# use float64'
from re import U

import matplotlib.pyplot as plt
from git import diff
from jax import config

config.update("jax_enable_x64", True)

import json
import logging
import os
import time
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import equinox as eqx
import git
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from configs import *
from jax import vmap

from hybrid_models.hh import (
    HH,
    integrate,
    scaled_integrate,
)
from hybrid_models.hh.channels import *
from hybrid_models.optimizers import (
    hybrid_optimizer,
    mask_gradients,
    jacobian_penalty_hutchinson,
)
from hybrid_models.transforms import (
    ModelParamTransform,
    ParamTransform,
)
from hybrid_models.utils import (
    BoxUniform,
    CustomJSONEncoder,
    DataLoader,
    Dataset,
    ProgressBar,
    assert_finite,
    fmt_elapsed_time,
    label_params,
    setup_logging,
    tree_filter_by_path,
    tree_path_of_leaves,
    tree_set_with_path,
    flatten_dict,
    nest_flattened_dict,
    update_config,
)

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
    default=None,
    help="config name",
)

args, unknown_args = parser.parse_known_args()
if args.config is None:
    raise ValueError("required argument --config is not set")
else:
    config = globals()[args.config]()

flat_config = flatten_dict(config.__dict__)
for k, v in flat_config.items():
    if isinstance(v, (list, tuple)):
        elem_type = type(v[0]) if len(v) > 0 else str
        parser.add_argument(
            f"--{k}",
            type=elem_type,
            default=v,
            help=k,
            nargs='+'
        )
    else:
        parser.add_argument(f"--{k}", type=type(v), default=v, help=k)

args = parser.parse_args()
nested_inputs = nest_flattened_dict(vars(args))
update_config(config, nested_inputs)

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

    if not args.no_fout:
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

    ## Setup ##
    rng = jr.key(config.seed)
    rng_keys = jr.split(rng, 10)

    # load training data
    logging.info(f"loading training and validation data")
    train_fpath = config.train_fpath
    val_fpath = config.val_fpath
    data_train = Dataset.from_file(train_fpath)

    if val_fpath is not None:
        data_val = Dataset.from_file(val_fpath)
    else:
        # If no validation file specified, use training data for validation
        data_val = deepcopy(data_train)

    observed_states = [k for k in data_train.Y.keys() if k != "time"]
    bounds = config.param_bounds

    # Extract time arrays once (all samples share the same time grid)
    ts_train = data_train.Y.pop("time")[0]  # Get time array from first sample
    ts_val = data_val.Y.pop("time")[0]  # Get time array from first sample
    
    # add observation noise
    noisy_Y = jax.tree.map(
        lambda x: x + config.obs_noise * jnp.max(jnp.abs(x)) * jr.normal(rng_keys[0], x.shape),
        data_train.Y
    )
    data_train.Y = noisy_Y

    logging.info(f"initializing model")
    trainable_params = config.trainable_params

    hh = config.hh_model

    @eqx.filter_vmap(in_axes=(None, 0, None, None, None, None))
    def v_integrate(model, params, u0, ts, observed_states=None, integrate_kwargs=None):
        _model = model.set(params)  # this should only set static params!
        _u0 = {**u0, **_model.initial_state}
        t, x = scaled_integrate(
            _model, ts, _u0, save_dims=observed_states, **integrate_kwargs
        )
        return t, x

    logging.info(f"initializing synthetic channel")
    # Initialize model and optimizer
    channel_kwargs = config.channel_model_kwargs

    init_params = config.init_params
    # add noise proportional to the magnitude of the initial parameter values
    if config.param_noise > 0.0:
        init_params = {
            k: v + config.param_noise * jnp.abs(v) * jr.normal(rng_keys[2], ())
            for k, v in init_params.items()
        }

    init_model = hh.set(init_params)  # init params with true values
    initial_state = init_model.init(0.0, config.u_init)
    
    # filter for existing states; needed for latent state selection
    state_names = init_model.init(0.0, {}).keys()
    initial_state = {k: v for k, v in initial_state.items() if k in state_names}

    init_model = eqx.tree_at(lambda x: x.initial_state, init_model, initial_state)
    if config.from_pretrained is not None:
        logging.info(f"loading pretrained model from {config.from_pretrained}")
        pretrained_model = eqx.tree_deserialise_leaves(config.from_pretrained, init_model)
        init_model = pretrained_model

    logging.info(f"initializing parameter transform")
    tf = ParamTransform(
        {k: config.param_transform(l, u) for k, (l, u) in bounds.items()}
    )
    tf = ModelParamTransform(tf)

    # if observed states exist on different scales (i.e. voltage and gates are both observed)
    y_scale = {k: jnp.maximum(jnp.std(data_train.Y[k]), 1e-3) for k in observed_states}

    @eqx.filter_jit
    def loss_fn(diff_model, params, u0, static_model, ti, yi, key, loss_kwargs=None, integrate_kwargs=None):
        """Compute negative log likelihood loss assuming Gaussian noise with time-varying uncertainty"""
        model = eqx.combine(diff_model, static_model)
        _model = tf.forward(model)
        ts_pred, y_pred_full = v_integrate(_model, params, u0, ti, None, integrate_kwargs)
        y_pred = {k: v for k, v in y_pred_full.items() if k in observed_states}

        # mse loss
        residuals = jax.tree.map(lambda x, y, s=1.0: (x - y) / s, y_pred, yi)
        flat_residuals, _ = jax.flatten_util.ravel_pytree(residuals)
        mse = jnp.mean(flat_residuals**2)
        
        state_targets = loss_kwargs.get("force_state", None)
        if state_targets is not None:
            y_pred_partial = {k: v for k, v in y_pred_full.items() if k in state_targets}
            residuals = jax.tree.map(lambda x, y: (x - y) / jnp.maximum(jnp.max(y), 1.0), y_pred_partial, state_targets)
            flat_residuals, _ = jax.flatten_util.ravel_pytree(residuals)
            mse = jnp.mean(flat_residuals**2)

        def time_grad(y, t):
            return jnp.gradient(y, t, axis=-1)
        lam1 = loss_kwargs.get("lam_dx", 1e-2)
        mse_dx = jnp.mean((time_grad(y_pred["v"], ti) - time_grad(yi["v"], ti))**2)

        # # --- Jacobian penalty ---
        # NOTE: This makes the latents look bad
        # K = 8
        # lam2 = 1e-3

        # B, T = ts_pred.shape[:2]
        # idx_t = jnp.linspace(0, T - 1, K, dtype=jnp.int32)
        # keys = jr.split(key, B * K).reshape(B, K)

        # # Pre-slice at sampled time points: shapes (B, K) and (B, K, ...)
        # ts_sampled = ts_pred[:, idx_t]
        # state_keys = ["v"] + _model.channels["node"].latent_states
        # u_sampled = {name: arr[:, idx_t] for name, arr in y_pred_full.items()}

        # # vmap over K time samples per batch element, then over B batch elements
        # jac_pen = jnp.mean(
        #     jax.vmap(jax.vmap(jacobian_penalty_hutchinson, (None, 0, 0, 0, None)), (None, 0, 0, 0, None))
        #     (_model, ts_sampled, u_sampled, keys, state_keys)
        # )
        loss = mse + lam1 * mse_dx # + lam2 * jac_pen
        return loss

    @eqx.filter_jit
    def make_step(
        diff_model, params, u0, static_model, ti, yi, opt_state, key, mask_grads=None, loss_kwargs=None, integrate_kwargs=None
    ):
        """Single optimization step using MLE (negative log likelihood)"""
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
            diff_model, params, u0, static_model, ti, yi, key, loss_kwargs, integrate_kwargs
        )

        grads = grads if mask_grads is None else mask_gradients(grads, mask_grads)
        # _loss_fn = lambda dm: loss_fn(dm[0], params, u0, static_model, ti, yi, key) # for lbfgs
        updates, opt_state = optimizer.update(
            [grads], opt_state, [diff_model], value=loss_val, #value_fn=_loss_fn, grad=[grads] # for lbfgs
        )

        updated_diff_model = eqx.apply_updates(diff_model, updates[0])
        # # prevent weight decay from causing masked weights to drift away from zero
        # updated_diff_model = zero_masked_weights(updated_diff_model)
        return updated_diff_model, opt_state, loss_val, grads

    # Prepare model for training
    logging.info(f"preparing model for training")

    # set trainable params and ignore static params
    filter_spec = jax.tree_util.tree_map(lambda x: False, init_model)
    if "net" in trainable_params:

        def mask_weight_or_bias(path, leaf):
            param_names = ["weight", "bias"]
            return any(name in str(p) for name in param_names for p in path)

        filter_spec = jax.tree_util.tree_map_with_path(mask_weight_or_bias, filter_spec)

    filter_spec = tree_set_with_path(filter_spec, {k: True for k in trainable_params})

    model = tf.inverse(init_model)
    diff_model, static_model = eqx.partition(model, filter_spec)

    # check if parameters are finite after transformation
    tf_model = tf.inverse(init_model)
    tf_model = tf.forward(tf_model)
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

    train_dataloader = DataLoader(
        data_train,
        shuffle=True,
        batch_size=config.batch_size,
        key=rng_keys[3],
        cycle_batches=True,
    )

    val_dataloader = DataLoader(
        data_val,
        shuffle=True,
        batch_size=config.batch_size,
        key=rng_keys[4],
        cycle_batches=True,
    )

    losses = {"train": [], "val": []}
    model_checkpoints = []
    logging.info(f"starting training")
    start_time = time.time()  # Start timing
    for epoch, strat in enumerate(config.opt_strategy, start=1):
        epoch_rng_key = jr.fold_in(rng_keys[5], epoch)

        # wrap in list for multi_transform compatibility
        # see https://github.com/patrick-kidger/equinox/issues/794
        ode_transforms = [optax.zero_nans(), optax.clip_by_global_norm(0.1), strat["ode_optim"]]
        optimizer = optax.chain(*ode_transforms)
        opt_state = optimizer.init([diff_model])

        best_model = {
            "step": 0,
            "train_loss": jnp.inf,
            "val_loss": jnp.inf,
            "model": None,
        }

        # Determine time array lengths for this epoch
        length_size = len(ts_train)
        frac_len = int(length_size * strat["length_frac"])
        ts_train_frac = ts_train[:frac_len]

        epoch_start_time = time.time()
        strat_integrate_kwargs = {**config.integrate_kwargs, **strat.get("integrate_kwargs", config.integrate_kwargs)}
        strat_loss_kwargs = strat.get("loss_kwargs", {})
        for step, (train_batch, val_batch) in enumerate(
            zip(train_dataloader, val_dataloader), start=1
        ):
            step_rng_key = jr.fold_in(epoch_rng_key, step)
            params_train, xs_train = train_batch
            params_val, xs_val = val_batch

            # Truncate training data to fraction length
            xs_train_frac = jax.tree.map(lambda x: x[:, :frac_len], xs_train)

            # needs to run before model is updated in make_step
            # validate on all full length data
            val_loss = jnp.nan
            if config.val_fpath is not None:
                val_loss = loss_fn(
                    diff_model, params_val, initial_state, static_model, ts_val, xs_val, step_rng_key, loss_kwargs=strat_loss_kwargs, integrate_kwargs=strat_integrate_kwargs
                )   

            updated_diff_model, opt_state, train_loss, grad = make_step(
                diff_model,
                params_train,
                initial_state.copy(),
                static_model,
                ts_train_frac,
                xs_train_frac,
                opt_state,
                step_rng_key,
                mask_grads=strat["mask_grads"],
                loss_kwargs=strat_loss_kwargs,
                integrate_kwargs=strat_integrate_kwargs,
            )

            losses["val"].append(val_loss)
            losses["train"].append(train_loss)

            if val_loss < best_model["val_loss"] or (jnp.isnan(val_loss) and train_loss < best_model["train_loss"]):
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
                }
            )

            val_errors = {k: [] for k in observed_states}
            val_kwargs = strat.get("val_kwargs", {})
            t_max = val_kwargs.get("t_frac", 1.0) * ts_val[-1]
            ts_debug = ts_val if t_max is None else ts_val[ts_val <= t_max]
            xs_debug = jax.tree.map(lambda x: x[:, : len(ts_debug)], xs_val)
            ts, xs_pred = v_integrate(
                model_checkpoint, params_val, initial_state, ts_debug, None, config.integrate_kwargs
            )
            map_mse = lambda x, y: jax.tree.map(
                lambda x, y: jnp.mean((x - y) ** 2), x, y
            )
            for k in observed_states:
                val_errors[k].append(map_mse(xs_debug[k], xs_pred[k]))
            debug_data = {"validation_data": {"ts": ts_debug, "val": xs_debug, "pred": xs_pred}}
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

    kwargs_str = channel_kwargs.copy()
    if "activation" in kwargs_str:
        kwargs_str["activation"] = kwargs_str["activation"].__name__
    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs_str.items()])

    model_import_instructions = f"""
The trained model has been saved to "{fout}.eqx" and can be imported using `eqx.tree_deserialise_leaves`.

```
hh = {hh.__repr__()}
loaded_model = eqx.tree_deserialise_leaves("{fout}.eqx", hh)
u0 = loaded_model.initial_state
```
"""
    if not args.no_fout:
        logging.info(model_import_instructions)


if __name__ == "__main__":
    main(config)
