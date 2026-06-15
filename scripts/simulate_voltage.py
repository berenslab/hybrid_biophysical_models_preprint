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

from copy import deepcopy
from argparse import ArgumentParser
from datetime import datetime

from hybrid_models.utils import (
    BoxUniform,
    CustomJSONEncoder,
    setup_logging,
    Dataset,
)
from hybrid_models.hh import (
    HH,
    integrate,
)
from hybrid_models.hh.channels import *

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

for k, v in config.__dict__.items():
    parser.add_argument(f"--{k}", type=type(v), default=v, help=k)

# overwrite config with args that are different from the default
args = parser.parse_args()
for k, v in config.__dict__.items():
    arg_v = getattr(args, k, v)
    if arg_v != v:
        setattr(config, k, arg_v)
    if k == "observed_states" and isinstance(config.observed_states, str):
        config.observed_states = config.observed_states.replace(" ", "").split(
            ","
        )  # convert to list


def main(config):
    if not os.path.exists(config.output_dir) and not args.no_fout:
        os.makedirs(f"{config.output_dir}")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    fout = f"{config.output_dir}/{config.fname}"

    ## HEADER ##
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    patch = repo.git.diff(":!*.ipynb") + "\n "  # "/n " fixes corrupt patch error

    # save config
    if not args.no_fout:
        data = deepcopy(config.__dict__)
        data["timestamp"] = timestamp
        data["commit"] = sha
        data["patch"] = patch  # add patch add the end of the file

        with open(f"{fout}_config.json", "w") as f:
            json.dump(data, f, cls=CustomJSONEncoder, indent=2)

    ## Set up logging
    setup_logging(f"{fout}.log", to_fout=not args.no_fout, to_stdout=args.verbose)

    # print config
    logging.info(
        f"""CONFIGURATION
experiment: {config.fname} @ {timestamp}
{"\n".join([f"{k}: {v}" for k, v in config.__dict__.items()])}
commit: {sha}
"""
    )

    ## Setup ##
    dt = config.dt
    ts = jnp.arange(0.0, config.tmax, dt)
    rng = jr.key(config.seed)
    rng_keys = jr.split(rng, 10)

    @eqx.filter_vmap(in_axes=(None, 0, None, None, None))
    def v_integrate(model, params, ts, initial_state, observed_states=None):
        model = model.set(params)  # this should only set static params!
        t, x = integrate(
            model,
            ts,
            initial_state,
            save_dims=observed_states,
            max_steps=config.solver_max_steps,
        )
        return t, x

    logging.info(f"initializing model")
    p_static = config.static_params

    hh = config.hh_model
    hh = hh.set(p_static)

    batch_items = list(config.batch_over.items())
    rng_keys_4_sampling = jr.split(rng_keys[1], 2 * len(batch_items))

    p_train = {
        k: sample_fn(rng_keys_4_sampling[i], config.num_train)
        for i, (k, sample_fn) in enumerate(batch_items)
    }
    p_val = {
        k: sample_fn(rng_keys_4_sampling[len(batch_items) + i], config.num_val)
        for i, (k, sample_fn) in enumerate(batch_items)
    }

    add_noise = lambda rng, std: lambda x: x + jr.normal(rng, x.shape) * std

    logging.info(f"Generate training data")
    initial_state = hh.init(0.0, config.u_init)

    t_train, y_train = v_integrate(
        hh, p_train, ts, initial_state, config.observed_states
    )
    y_train = jax.tree.map(add_noise(rng_keys[2], config.train_noise_std), y_train)
    y_train = {"time": t_train, **y_train}

    logging.info(f"generating validation data")
    t_val, y_val = v_integrate(hh, p_val, ts, initial_state, config.observed_states)
    y_val = jax.tree.map(add_noise(rng_keys[5], config.val_noise_std), y_val)
    y_val = {"time": t_val, **y_val}

    data_train = Dataset({**p_train, "model_params": p_static}, y_train)
    data_val = Dataset({**p_val, "model_params": p_static}, y_val)

    data_train.to_file(f"{fout}_train.json")
    data_val.to_file(f"{fout}_val.json")


if __name__ == "__main__":
    main(config)
