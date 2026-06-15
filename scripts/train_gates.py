# use float64'
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import git
import os
import json

import equinox as eqx
import optax

from copy import deepcopy
from argparse import ArgumentParser
from datetime import datetime
import time

from hybrid_models.utils import (
    BoxUniform,
    ProgressBar,
    Dataset,
    DataLoader,
    tree_filter_by_path,
    CustomJSONEncoder,
    setup_logging,
    tree_set_with_path,
    tree_path_of_leaves,
    fmt_elapsed_time,
    assert_finite,
)
from hybrid_models.transforms import (
    ParamTransform,
    ModelParamTransform,
)
from hybrid_models.hh import (
    HH,
    integrate,
)
from hybrid_models.hh.channels import *

from hybrid_models.optimizers import (
    hybrid_optimizer,
    mask_gradients,
)

from configs import *
import logging

parser = ArgumentParser()

parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="print logging to console",
)
parser.add_argument(
    "--no_fout",
    action="store_true",
    default=False,
    help="no output files",
)
parser.add_argument(
    "--no_weights",
    action="store_true",
    default=False,
    help="no weights saving",
)
parser.add_argument(
    "--config",
    type=str,
    default=train_node_on_gates,
    help="config name",
)

args, unknown_args = parser.parse_known_args()
if args.config is None:
    raise ValueError("required argument --config is not set")
else:
    config = globals()[args.config]()

for k, v in config.__dict__.items():
    parser.add_argument(f"--{k}", type=type(v), default=v, help=k)

args = parser.parse_args()
for k, v in config.__dict__.items():
    arg_v = getattr(args, k, v)
    if arg_v != v:
        setattr(config, k, arg_v)


def main(config):
    if not os.path.exists(config.output_dir) and not args.no_fout:
        os.makedirs(f"{config.output_dir}")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    fout = f"{config.output_dir}/{config.exp_name}_{timestamp}"

    ## Experiment HEADER ##
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    patch = repo.git.diff(":!*.ipynb") + "\n "  # "/n " fixes corrupt patch error

    # save config
    if not args.no_fout:
        data = deepcopy(config.__dict__)
        data["timestamp"] = timestamp
        data["commit"] = sha
        data["patch"] = patch  # add patch add the end of the file
        for metric in [
            "final_params",
            "best_train_loss",
            "best_val_loss",
            "elapsed_time",
            "num_steps",
        ]:
            data[metric] = None

        with open(f"{fout}.json", "w") as f:
            json.dump(data, f, cls=CustomJSONEncoder, indent=2)

    ## Set up logging
    setup_logging(f"{fout}.log", to_fout=not args.no_fout, to_stdout=args.verbose)

    # print config
    logging.info(
        f"""CONFIGURATION
experiment: {config.exp_name}_{timestamp}
{"\n".join([f"{k}: {v}" for k, v in config.__dict__.items()])}
commit: {sha}
"""
    )

    start_time = time.time()  # Start timing

    ## Setup ##
    rng = jr.key(config.seed)
    rng_keys = jr.split(rng, 10)

    # load training data
    logging.info(f"loading training and validation data")

    data_train = Dataset.from_file(config.train_fpath)
    val_fpath = config.train_fpath if config.val_fpath is None else config.val_fpath
    data_val = Dataset.from_file(val_fpath)

    logging.info(f"initializing channel model")
    observed_states = list(data_train.X["v"].keys())
    init_channel_model = config.ChannelModel(
        key=rng_keys[3], **config.channel_model_kwargs, latent_states=observed_states
    )
    init_channel_model = init_channel_model.set(config.init_params)
    model_name = config.ChannelModel.__name__

    logging.info(f"initializing parameter transform")
    bounds = config.param_bounds
    tf = ParamTransform(
        {k: config.param_transform(l, u) for k, (l, u) in bounds.items()}
    )
    tf = ModelParamTransform(tf)

    @eqx.filter_jit
    def mse_loss(diff_model, static_model, voltage_data, xinf_true, tau_true):
        model = eqx.combine(diff_model, static_model)
        _model = tf.forward(model)
        xinf_pred, tau_pred = eqx.filter_vmap(_model._compute_gates)(
            0.0, {"v": voltage_data}, None
        )

        xinf_scale = jax.tree.map(
            lambda y: jnp.maximum(y.std(axis=-1, keepdims=True), 1.0), xinf_true
        )
        tau_scale = jax.tree.map(
            lambda y: jnp.maximum(y.std(axis=-1, keepdims=True), 1.0), tau_true
        )

        # xinf_scale = jax.tree.map(lambda y: jnp.max(y), xinf_true)
        # tau_scale = jax.tree.map(lambda y: jnp.max(y), tau_true)

        xinf_residual = jax.tree.map(
            lambda x, y, s=1.0: (x - y) / s, xinf_true, xinf_pred, xinf_scale
        )
        tau_residual = jax.tree.map(
            lambda x, y, s=1.0: (x - y) / s, tau_true, tau_pred, tau_scale
        )
        mse = (
            jax.tree_util.tree_reduce(
                lambda acc, res: acc + jnp.mean(res**2), xinf_residual, 0.0
            )
            + jax.tree_util.tree_reduce(
                lambda acc, res: acc + jnp.mean(res**2), tau_residual, 0.0
            )
        ) / 2

        return mse

    @eqx.filter_jit
    def make_step(
        diff_model,
        static_model,
        voltage_data,
        xinf_true,
        tau_true,
        opt_state,
        mask_grads=None,
    ):
        loss_val, grads = eqx.filter_value_and_grad(mse_loss)(
            diff_model, static_model, voltage_data, xinf_true, tau_true
        )

        grads = grads if mask_grads is None else mask_gradients(grads, mask_grads)
        updates, opt_state = optimizer.update(
            grads, opt_state, diff_model, value=loss_val
        )

        updated_diff_model = eqx.apply_updates(diff_model, updates)
        # # prevent weight decay from causing masked weights to drift away from zero
        # updated_diff_model = zero_masked_weights(updated_diff_model)
        return updated_diff_model, opt_state, loss_val, grads

    # Prepare model for training
    logging.info(f"preparing model for training")

    # ignore static params
    filter_spec = jax.tree_util.tree_map(lambda x: False, init_channel_model)
    filter_spec = tree_set_with_path(
        filter_spec, {k: True for k in config.trainable_params}
    )
    if "net" in config.trainable_params:

        def mask_weight_or_bias(path, leaf):
            return any("weight" in str(p) or "bias" in str(p) for p in path)

        filter_spec = jax.tree_util.tree_map_with_path(mask_weight_or_bias, filter_spec)

    channel_model = tf.inverse(init_channel_model)
    diff_model, static_model = eqx.partition(channel_model, filter_spec)

    # check if parameters are finite after transformation
    tf_model = tf.forward(init_channel_model)
    assert_finite(tf_model, data_train.X)
    assert_finite(tf_model, data_val.X)

    # Training loop
    pbar = ProgressBar(
        fmt="Epoch {epoch}/"
        + f"{len(config.opt_strategy)}"
        + ": {self} {step}/{num_steps}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}"
        + " " * 17
    )
    logging.info(f"setting up dataloaders")

    losses = {"train": [], "val": []}
    model_checkpoints = []
    logging.info(f"starting training")

    train_dataloader = DataLoader(
        data_train,
        shuffle=True,
        batch_size=config.batch_size,
        key=rng_keys[5],
        cycle_batches=True,
    )

    val_dataloader = DataLoader(
        data_val,
        shuffle=True,
        batch_size=config.batch_size,
        key=rng_keys[6],
        cycle_batches=True,
    )
    for epoch, strat in enumerate(config.opt_strategy, start=1):
        epoch_rng_key = jr.fold_in(rng_keys[7], epoch)
        optimizer = strat["optimizer"]

        opt_state = optimizer.init(diff_model)

        best_model = {
            "step": 0,
            "train_loss": jnp.inf,
            "val_loss": jnp.inf,
            "model": None,
        }
        epoch_start_time = time.time()

        for step, (train_batch, val_batch) in enumerate(
            zip(train_dataloader, val_dataloader), start=1
        ):
            xs_train, ys_train = train_batch
            xs_val, ys_val = val_batch
            v_data_train = jax.tree.leaves(xs_train["v"])[0]

            xinf_data_train = ys_train["xinf"]
            tau_data_train = ys_train["tau"]

            v_data_val = jax.tree.leaves(xs_val["v"])[0]
            xinf_data_val = ys_val["xinf"]
            tau_data_val = ys_val["tau"]

            # needs to run before model is updated in make_step
            # validate on all full length data
            val_loss = mse_loss(
                diff_model, static_model, v_data_val, xinf_data_val, tau_data_val
            )

            updated_diff_model, opt_state, train_loss, grad = make_step(
                diff_model,
                static_model,
                v_data_train,
                xinf_data_train,
                tau_data_train,
                opt_state,
            )

            losses["val"].append(val_loss)
            losses["train"].append(train_loss)

            if val_loss < best_model["val_loss"]:
                best_model = {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                model_checkpoint = deepcopy(
                    tf.forward(eqx.combine(diff_model, static_model))
                )
                best_model["model"] = model_checkpoint

            progress = pbar(
                step / strat["max_steps"],
                epoch=epoch,
                train_loss=train_loss,
                step=step,
                num_steps=strat["max_steps"],
                val_loss=val_loss,
            )

            print(progress, end="\r")
            if step % 100 == 0 or step == 1:
                logging.info(progress)

            diff_model = updated_diff_model  # update diff_model for next step

            # max_steps needed to avoid infinite loop when cycle_batches is True
            if step >= strat["max_steps"] or step - best_model["step"] > 10_000:
                print()
                break
        model_checkpoints.append(model_checkpoint)

        # Epoch summary
        epoch_params = tree_filter_by_path(model_checkpoint, config.trainable_params)
        flat_with_paths = jax.tree_util.tree_flatten_with_path(epoch_params)[0]
        epoch_params = {
            "".join([str(p) for p in path]): leaf
            for path, leaf in list(flat_with_paths)
            if leaf is not None
        }
        epoch_dur_str = fmt_elapsed_time(time.time() - epoch_start_time)
        elapsed_so_far = fmt_elapsed_time(time.time() - start_time)
        num_steps_so_far = len(losses["train"])
        last_train_loss = losses["train"][-1]
        last_val_loss = losses["val"][-1]
        top_val_loss = best_model["val_loss"]

        logging.info(f"epoch {epoch} summary:")
        logging.info(
            f"  train loss: {last_train_loss:.4f}, val loss: {last_val_loss:.4f}"
        )
        logging.info(f"  top val loss: {top_val_loss:.4f}")
        logging.info(f"  estimated params: {epoch_params}")
        logging.info(f"  epoch duration: {epoch_dur_str}")
        logging.info(f"  in total: {elapsed_so_far}s for {num_steps_so_far} steps")

        if not args.no_fout:
            with open(f"{fout}.json", "r") as f:
                data = json.load(f)
            data.update(
                {
                    "final_params": epoch_params,
                    "best_train_loss": last_train_loss,
                    "best_val_loss": top_val_loss,
                    "elapsed_time": elapsed_so_far,
                    "num_steps": num_steps_so_far,
                    "train_losses": losses["train"],
                    "val_losses": losses["val"],
                    "observed_states": observed_states,
                }
            )

            eval_dataloader = DataLoader(
                data_val,
                shuffle=False,
                batch_size=config.batch_size,
                key=rng_keys[6],
                cycle_batches=False,
            )
            val_errors = {"xinf": [], "tau": []}
            for i, (val_batch) in enumerate(eval_dataloader):
                xs_val, ys_val = val_batch
                xinf_data_val = ys_val["xinf"]
                tau_data_val = ys_val["tau"]
                v_data_val = jax.tree.leaves(xs_val["v"])[0]
                xinf_pred_val, tau_pred_val = eqx.filter_vmap(
                    model_checkpoint._compute_gates
                )(0.0, {"v": v_data_val}, None)
                map_mse = lambda x, y: jax.tree.map(
                    lambda x, y: jnp.mean((x - y) ** 2), x, y
                )
                val_errors["xinf"].append(map_mse(xinf_data_val, xinf_pred_val))
                val_errors["tau"].append(map_mse(tau_data_val, tau_pred_val))

                if i == 0:
                    ys_pred = {"xinf": xinf_pred_val, "tau": tau_pred_val}
                    debug_data = {
                        "validation_data": {
                            "v": v_data_val,
                            "val": ys_val,
                            "pred": ys_pred,
                        }
                    }
                    data.update(debug_data)

            val_errors = {
                k: jnp.mean(jax.flatten_util.ravel_pytree(v)[0])
                for k, v in val_errors.items()
            }
            data["validation_error"] = val_errors

            with open(f"{fout}.json", "w") as f:
                json.dump(data, f, cls=CustomJSONEncoder, indent=2)

            if not args.no_weights:
                eqx.tree_serialise_leaves(f"{fout}.eqx", model_checkpoint)

    elapsed_time = time.time() - start_time
    elapsed_str = fmt_elapsed_time(elapsed_time)

    num_steps = len(losses["train"])
    logging.info(f"training finished after {num_steps} steps and took {elapsed_str}.")

    trained_model = model_checkpoints[-1]
    if not args.no_fout and not args.no_weights:
        eqx.tree_serialise_leaves(f"{fout}.eqx", trained_model)

    kwargs_str = config.channel_model_kwargs.copy()
    kwargs_str["latent_states"] = observed_states
    if "activation" in kwargs_str:
        kwargs_str["activation"] = kwargs_str["activation"].__name__
    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs_str.items()])

    model_import_instructions = f"""
The trained model has been saved to "{fout}.eqx" and can be imported using `eqx.tree_deserialise_leaves`.

```
channel_model = {model_name}(key=jr.key(0), {kwargs_str})
loaded_channel_model = eqx.tree_deserialise_leaves("{fout}.eqx", channel_model)
```
"""
    if not args.no_fout:
        logging.info(model_import_instructions)


if __name__ == "__main__":
    main(config)
