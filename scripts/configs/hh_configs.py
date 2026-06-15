from dataclasses import dataclass, field
from typing import Optional

import diffrax

from hybrid_models.utils import absorb_kwargs, custom_initializer
from hybrid_models.hh.channels import *
from hybrid_models.hh.base import ExternalInput, StepCurrent
from hybrid_models.transforms import (
    LogisticTransform,
    IdentityTransform,
    SigmoidTransform,
)
from hybrid_models.optimizers import preconditioned_sgld, sgld, hybrid_optimizer

from hybrid_models.hh import HH
import optax
import jax
import jax.numpy as jnp
import jax.random as jr


@dataclass
class generate_hh_single_spike:
    output_dir: str = "data/hh_synth"
    seed: int = 0
    tmax: float = 25.0
    dt: float = 0.1
    observed_states: str = "v"
    num_train: int = 1
    num_val: int = 1
    fname: str = f"hh_single_ap_v_only"
    train_noise_std: float = 0.0
    val_noise_std: float = 0.0
    u_init = {"v": jnp.array(-70.0)}
    solver_max_steps: int = 10_000

    hh_model: HH = HH([Na(), K(), Leak()], [StepCurrent(amp=10.0, start=5.0, end=25.0)])

    static_params: dict = field(
        default_factory=lambda: {
            ".r": jnp.array(1.0),  # radius
            ".l": jnp.array(10.0),  # length
            ".c": jnp.array(1.0),  # capacitance
            ".tadj": jnp.array(
                3.0 ** ((37.0 - 37.0) / 10.0)
            ),  # Q10 temperature adjustment factor
            ".channels['leak'].gl": jnp.array(0.3),  # leak conductance
            ".channels['leak'].el": jnp.array(-70.0),  # leak reversal potential
            ".channels['leak'].tadj": jnp.array(3.0 ** ((37.0 - 37.0) / 10.0)),
            ".channels['na'].gna": jnp.array(120.0),  # sodium conductance
            ".channels['na'].ena": jnp.array(50.0),  # sodium reversal potential
            ".channels['na'].tadj": jnp.array(3.0 ** ((37.0 - 37.0) / 10.0)),
            ".channels['k'].gk": jnp.array(36.0),  # potassium conductancea
            ".channels['k'].ek": jnp.array(-75.0),  # potassium reversal potential
            ".channels['k'].tadj": jnp.array(3.0 ** ((37.0 - 37.0) / 10.0)),
        }
    )

    batch_over: dict = field(
        default_factory=lambda: {
            # ".c": lambda key, n: jnp.linspace(0.5, 1.5, n),
            # ".r": lambda key, n: jnp.linspace(0.5, 1.5, n),
            # ".l": lambda key, n: jnp.linspace(1.0, 20.0, n),
            # ".tadj": lambda key, n: jnp.linspace(1.0, 1.5, n),
            # ".channels['k'].gk": lambda key, n: jnp.linspace(20.0, 40.0, n),
            # ".channels['k'].ek": lambda key, n: jnp.linspace(-90.0, -50.0, n),
            # ".channels['leak'].gl": lambda key, n: jnp.linspace(0.01, 0.5, n),
            # ".channels['leak'].el": lambda key, n: jnp.linspace(-80.0, -60.0, n),
            # ".channels['na'].gna": lambda key, n: jnp.linspace(100.0, 200.0, n),
            # ".channels['na'].ena": lambda key, n: jnp.linspace(30.0, 60.0, n),
            # ".externals['i_ext'].amp": lambda key, n: jnp.linspace(-5.0, 15.0, n),
            # ".externals['i_ext'].start": lambda key, n: jnp.linspace(0.0, 10.0, n),
            # ".externals['i_ext'].end": lambda key, n: jnp.linspace(15.0, 25.0, n),

            # ".c": lambda key, n: jr.uniform(key, (n,), minval=0.5, maxval=1.5),
            # ".r": lambda key, n: jr.uniform(key, (n,), minval=0.5, maxval=1.5),
            # ".l": lambda key, n: jr.uniform(key, (n,), minval=1.0, maxval=20.0),
            # ".tadj": lambda key, n: jr.uniform(key, (n,), minval=1.0, maxval=1.5),
            # ".channels['k'].gk": lambda key, n: jr.uniform(key, (n,), minval=20.0, maxval=40.0),
            # ".channels['k'].ek": lambda key, n: jr.uniform(key, (n,), minval=-90.0, maxval=-50.0),
            # ".channels['leak'].gl": lambda key, n: jr.uniform(
            #     key, (n,), minval=0.01, maxval=0.5
            # ),
            # ".channels['leak'].el": lambda key, n: jr.uniform(
            #     key, (n,), minval=-80.0, maxval=-60.0
            # ),
            # ".channels['na'].gna": lambda key, n: jr.uniform(
            #     key, (n,), minval=100.0, maxval=200.0
            # ),
            # ".channels['na'].ena": lambda key, n: jr.uniform(
            #     key, (n,), minval=30.0, maxval=60.0
            # ),
            # ".externals['i_ext'].amp": lambda key, n: jr.uniform(
            #     key, (n,), minval=-5.0, maxval=15.0
            # ),
            # ".externals['i_ext'].start": lambda key, n: jr.uniform(
            #     key, (n,), minval=0.0, maxval=10.0
            # ),
            # ".externals['i_ext'].end": lambda key, n: jr.uniform(
            #     key, (n,), minval=15.0, maxval=25.0
            # ),
        }
    )

@dataclass
class generate_hh_multi_spike(generate_hh_single_spike):
    tmax: float = 50.0
    fname: str = f"hh_multi_ap_v_only"
    hh_model: HH = HH([Na(), K(), Leak()], [StepCurrent(amp=15.0, start=5.0, end=45.0)])
    
@dataclass
class generate_hh_multi_spike_init(generate_hh_multi_spike):
    fname: str = f"hh_multi_ap_v_only_init"
    u_init = {"v": jnp.array(-70.0), "m": jnp.array(0.1), "n": jnp.array(0.1), "h": jnp.array(0.1)}

@dataclass
class generate_hh_multi_spike_batch(generate_hh_multi_spike):
    fname: str = f"hh_multi_ap_batch_v_only"
    num_train: int = 21
    batch_over: dict = field(
        default_factory=lambda: {
                    ".externals['i_ext'].amp": lambda key, n: jnp.linspace(-10.0, 30.0, n),
        }
    )

@dataclass
class train_hybrid_on_hh:
    output_dir: str = "."
    seed: int = 1
    batch_size: int = 1
    ChannelModel: Channel = BioPhysicsNODE1
    exp_name: str = f"train_channel"
    param_noise: float = 0.0
    obs_noise: float = 0.0
    train_fpath: str = "data/hh_synth/hh_multi_ap_v_only_train.json"
    val_fpath: Optional[str] = "data/hh_synth/hh_multi_ap_v_only_val.json"
    from_pretrained: Optional[str] = None

    param_transform: callable = LogisticTransform

    channel_model_kwargs: dict = field(
        default_factory=lambda: {}
    )

    hh_model: HH = HH([Na(), K(), Leak()], [StepCurrent(amp=15.0, start=5.0, end=45.0)])

    u_init: dict = field(
        default_factory=lambda: {
            "v": jnp.array(-70.0),
            **Na().init(0.0, {"v": jnp.array(-70.0)}),
            **K().init(0.0, {"v": jnp.array(-70.0)}),
        }
    )

    integrate_kwargs: dict = field(
        default_factory=lambda: {
            "max_steps": 50_000,
            "rtol": 1e-6,
            "atol": 1e-8,
            "solver": diffrax.Tsit5(),
            "adjoint": diffrax.RecursiveCheckpointAdjoint(),
        }
    )

    init_params: dict = field(
        default_factory=lambda: {
        }
    )

    param_bounds: dict = field(
        default_factory=lambda: {
            ".c": (0.5, 1.5),
            ".r": (0.5, 1.5),
            ".l": (1.0, 20.0),
            ".channels['k'].gk": (20.0, 40.0),
            ".channels['k'].ek": (-90.0, -50.0),
            ".channels['leak'].gl": (0.01, 0.5),
            ".channels['leak'].el": (-100.0, -10.0),
            ".channels['na'].gna": (100.0, 200.0),
            ".channels['na'].ena": (30.0, 60.0),
            ".channels['node'].gx": (10.0, 200.0),
            ".channels['node'].ex": (-100.0, 100.0),
            ".channels['node'].powx['m']": (2.5, 3.5),
            ".channels['node'].powx['h']": (0.5, 1.5),
            ".channels['node'].powx['n']": (3.5, 5.5),
            ".initial_state['m']": (0.0, 1.0),
            ".initial_state['h']": (0.0, 1.0),
            ".initial_state['n']": (0.0, 1.0),
        }
    )

@dataclass
class train_node_on_hh:
    output_dir: str = "."
    seed: int = 1
    batch_size: int = 1
    ChannelModel: Channel = NODE
    exp_name: str = f"train_node_only"
    obs_noise: float = 0.0
    train_fpath: str = "data/hh_synth/hh_multi_ap_v_only_train.json"
    val_fpath: Optional[str] = "data/hh_synth/hh_multi_ap_v_only_val.json"
    from_pretrained: Optional[str] = None

    channel_model_kwargs: dict = field(
        default_factory=lambda: {
            "width_size": 64,
            "depth_size": 3,
            "activation": jax.nn.tanh,
            "latent_states": ["z0", "z1", "z2"],
            "last_layer_initializer": jax.nn.initializers.normal(stddev=0.1),
            "return_current": False,
        }
    )

    u_init: dict = field(
        default_factory=lambda: {
            "v": jnp.array(-70.0),
            **{f"z{i}": 0.0*jr.normal(jr.PRNGKey(i+42)) for i in range(3)},
        }
    )

    integrate_kwargs: dict = field(
        default_factory=lambda: {
            "max_steps": 50_000,
            "rtol": 1e-6,
            "atol": 1e-8,
            "solver": diffrax.Tsit5(),
            "adjoint": diffrax.RecursiveCheckpointAdjoint(),
        }
    )

    trainable_params: list = field(
        default_factory=lambda: [
            "net",
            # ".initial_state['v']",
            # ".initial_state['z0']",
            # ".initial_state['z1']",
            # ".initial_state['z2']",
            # ".initial_state['z3']",
            # ".initial_state['z4']",
        ]
    )

    opt_strategy: list[dict] = field(
        default_factory=lambda: [
            {
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1e-1, 500),
                    weight_decay=1e-5,
                ),
                "max_steps": 500,
                "length_frac": 0.10,
                "mask_grads": None,
                "loss_kwargs": {
                    "force_state": {"v": jnp.array(-70.0), **{f"z{i}": 0.0 for i in range(3)}},
                },
            },
            {
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1e-4, 500),
                    weight_decay=1e-5,
                ),
                "max_steps": 500,
                "length_frac": 0.22,
                "mask_grads": None,
            },
            {
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-5),
                    weight_decay=1e-5,
                ),
                "max_steps": 1000,
                "length_frac": 1.0,
                "mask_grads": None,
            },
        ]
    )


@dataclass
class train_hybrid_on_hh_na(train_hybrid_on_hh):
    exp_name: str = f"train_na"

    channel_model_kwargs: dict = field(
        default_factory=lambda: {
            "width_size": 64,
            "depth_size": 3,
            "activation": jax.nn.tanh,
            "latent_states": ["m", "h"],
            "last_layer_initializer": jax.nn.initializers.normal(stddev=0.1),
        }
    )

    hh_model: HH = train_hybrid_on_hh.hh_model.delete("na")

    init_params: dict = field(
        default_factory=lambda: {
            ".c": jnp.array(1.0),  # capacitance
            ".r": jnp.array(1.0),
            ".l": jnp.array(10.0),
            ".channels['leak'].gl": jnp.array(0.3),  # leak conductance
            ".channels['leak'].el": jnp.array(-70.0),  # leak reversal potential
            ".channels['k'].gk": jnp.array(36.0),  # potassium conductancea
            ".channels['k'].ek": jnp.array(-75.0),  # potassium reversal potential
            ".channels['node'].gx": jnp.array(120.0),
            ".channels['node'].ex": jnp.array(50.0),
            ".channels['node'].powx['m']": jnp.array(3.0),
            ".channels['node'].powx['h']": jnp.array(1.0),
        }
    )

    trainable_params: list = field(
        default_factory=lambda: [
            "net",
            # ".initial_state['m']",
            # ".initial_state['h']",
            # ".initial_state['n']",
            # # NOTE: fix for testing
            # ".channels['node'].gx",
            # ".channels['node'].ex",
            # ".channels['node'].powx['m']",
            # ".channels['node'].powx['h']",
            # ".channels['node'].powx['n']",
        ]
    )

    opt_strategy: list[dict] = field(
        default_factory=lambda: [
            {
                "ode_optim": optax.adam(learning_rate=optax.constant_schedule(1e-2)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1e-1, 500),
                    weight_decay=1e-5,
                ),
                "max_steps": 500,
                "length_frac": 0.10,
                "mask_grads": None,
                "loss_kwargs": {
                    "force_state": {"v": jnp.array(-70.0), **Na().init(0.0, {"v": jnp.array(-70.0)})},
                },
            },
            {
                "ode_optim": optax.adam(learning_rate=optax.constant_schedule(1e-2)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1e-4, 500),
                    weight_decay=1e-5,
                ),
                "max_steps": 500,
                "length_frac": 0.30,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(learning_rate=optax.constant_schedule(1e-2)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-5),
                    weight_decay=1e-5,
                ),
                "max_steps": 1000,
                "length_frac": 1.0,
                "mask_grads": None,
            },
        ]
    )


@dataclass
class train_hybrid_on_hh_na_multi(train_hybrid_on_hh_na):
    exp_name: str = f"train_na_multi"
    train_fpath: str = "data/hh_synth/hh_multi_ap_batch_v_only_train.json"
    val_fpath: Optional[str] = "data/hh_synth/hh_multi_ap_batch_v_only_val.json"
    batch_size: int = 8


@dataclass
class train_hybrid_on_hh_k(train_hybrid_on_hh):
    exp_name: str = f"train_k"
    
    channel_model_kwargs: dict = field(
        default_factory=lambda: {
            "width_size": 32,
            "depth_size": 2,
            "activation": jax.nn.tanh,
            "use_layer_norm": True,
            "latent_states": ["n"],
            "last_layer_initializer": jax.nn.initializers.normal(stddev=0.1),

        }
    )

    hh_model: HH = train_hybrid_on_hh.hh_model.delete("k")

    init_params: dict = field(
        default_factory=lambda: {
            ".c": jnp.array(1.0),  # capacitance
            ".r": jnp.array(1.0),
            ".l": jnp.array(10.0),
            ".channels['leak'].gl": jnp.array(0.3),  # leak conductance
            ".channels['leak'].el": jnp.array(-70.0),  # leak reversal potential
            ".channels['na'].gna": jnp.array(120.0),  # sodium conductance
            ".channels['na'].ena": jnp.array(50.0),  # sodium reversal potential
            ".channels['node'].gx": jnp.array(36.0),
            ".channels['node'].ex": jnp.array(-75.0),
            ".channels['node'].powx['n']": jnp.array(4.0),
        }
    )

    trainable_params: list = field(
        default_factory=lambda: [
            "net",
            # ".initial_state['m']",
            # ".initial_state['h']",
            # ".initial_state['n']",
            # ".channels['node'].powx['m']",
            # ".channels['node'].powx['h']",
            # ".channels['node'].powx['n']",
            
            # # incl trainable biophysical params
            # ".channels['node'].gx",
            # # ".channels['node'].ex",

            # ".channels['leak'].gl",
            # ".channels['leak'].el",
            # ".channels['na'].gna",
            # # ".channels['na'].ena",
            # ".channels['k'].gk",
            # # ".channels['k'].ek",

        ]
    )

    opt_strategy: list[dict] = field(
        default_factory=lambda: [
            {
                "ode_optim": optax.adam(learning_rate=optax.constant_schedule(1e-2)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1e-1, 500),
                    weight_decay=1e-5,
                ),
                "max_steps": 500,
                "length_frac": 0.10,
                "mask_grads": None,
                "loss_kwargs": {
                    "force_state": {"v": jnp.array(-70.0), **K().init(0.0, {"v": jnp.array(-70.0)})},
                },
            },
            {
                "ode_optim": optax.adam(learning_rate=optax.constant_schedule(1e-2)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1e-4, 500),
                    weight_decay=1e-5,
                ),
                "max_steps": 500,
                "length_frac": 0.30,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(learning_rate=optax.constant_schedule(1e-2)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-5),
                    weight_decay=1e-5,
                ),
                "max_steps": 1000,
                "length_frac": 1.0,
                "mask_grads": None,
            },
        ]
    )


@dataclass
class train_hybrid_and_hh_k(train_hybrid_on_hh_k):
    init_params: dict = field(
        default_factory=lambda: {
            ".c": jnp.array(1.0),  # capacitance
            ".r": jnp.array(1.0),
            ".l": jnp.array(10.0),
            ".channels['leak'].gl": jr.uniform(jr.split(jr.PRNGKey(train_hybrid_on_hh_k.seed), 10)[0], minval=0.01, maxval=0.5),
            ".channels['leak'].el": jr.uniform(jr.split(jr.PRNGKey(train_hybrid_on_hh_k.seed), 10)[1], minval=-100.0, maxval=-10.0),
            ".channels['node'].gx": jr.uniform(jr.split(jr.PRNGKey(train_hybrid_on_hh_k.seed), 10)[3], minval=10.0, maxval=200.0),
            ".channels['na'].gna": jr.uniform(jr.split(jr.PRNGKey(train_hybrid_on_hh_k.seed), 10)[2], minval=100.0, maxval=200.0),
            ".channels['node'].ex": jnp.array(-75.0),
            ".channels['node'].powx['n']": jnp.array(4.0),
            ".channels['na'].ena": jnp.array(50.0),
        }
    )

    trainable_params: list = field(
        default_factory=lambda: [
            "net",
            # ".initial_state['n']",
            # ".channels['node'].powx['n']",
            
            # incl trainable biophysical params
            ".channels['node'].gx",
            ".channels['leak'].gl",
            ".channels['leak'].el",
            ".channels['na'].gna",
            ".channels['k'].gk",
        ]
    )

    opt_strategy: list[dict] = field(
        default_factory=lambda: [
            # {
            #     "ode_optim": optax.adam(learning_rate=optax.cosine_onecycle_schedule(1e-1, 500)),
            #     "nn_optim": optax.adamw(
            #         learning_rate=optax.cosine_onecycle_schedule(1e-3, 500),
            #         weight_decay=1e-5,
            #     ),
            #     "max_steps": 500,
            #     "length_frac": 0.05,
            #     "mask_grads": None,
            # },
            {
                "ode_optim": optax.adam(learning_rate=optax.constant_schedule(1e-1)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.constant_schedule(1e-3),
                    weight_decay=1e-5,
                ),
                "max_steps": 1000,
                "length_frac": 0.22,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(learning_rate=optax.constant_schedule(1e-1)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.constant_schedule(1e-5),
                    weight_decay=1e-5,
                ),
                "max_steps": 1000,
                "length_frac": 1.0,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(learning_rate=optax.cosine_decay_schedule(1e-2, 1000)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_decay_schedule(1e-5, 1000),
                    weight_decay=1e-5,
                ),
                "max_steps": 1000,
                "length_frac": 1.0,
                "mask_grads": None,
            },
        ]
    )

@dataclass
class train_hybrid_on_hh_k_batch(train_hybrid_on_hh_k):
    exp_name: str = f"train_k_batch"
    train_fpath: str = "data/hh_synth/hh_multi_ap_batch_v_only_train.json"
    val_fpath: Optional[str] = "data/hh_synth/hh_multi_ap_batch_v_only_val.json"
    batch_size: int = 21

    opt_strategy: list[dict] = field(
        default_factory=lambda: [
            {
                "ode_optim": optax.adam(learning_rate=optax.constant_schedule(1e-2)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1e-1, 500),
                    weight_decay=1e-5,
                ),
                "max_steps": 500,
                "length_frac": 0.10,
                "mask_grads": None,
                "loss_kwargs": {
                    "force_state": {"v": jnp.array(-70.0), **K().init(0.0, {"v": jnp.array(-70.0)})},
                },
            },
            {
                "ode_optim": optax.adam(learning_rate=optax.constant_schedule(1e-2)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1e-4, 500),
                    weight_decay=1e-5,
                ),
                "max_steps": 500,
                "length_frac": 0.22,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(learning_rate=optax.constant_schedule(1e-2)),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-5),
                    weight_decay=1e-5,
                ),
                "max_steps": 1000,
                "length_frac": 1.0,
                "mask_grads": None,
            },
        ]
    )

@dataclass
class train_hybrid_on_multicomp:
    output_dir: str = "."
    seed: int = 0
    batch_size: int = 1
    ChannelModel: Channel = NODE  # works best with mish activation and plain Adam
    # ChannelModel: Channel = BioPhysicsNODE2  # works best with mish activation and plain AdamW
    train_fpath: str = "data/hh_synth/pospischil_mutlicomp_single_pt_soma.json"
    obs_noise: float = 0.0
    param_noise: float = 0.0
    val_fpath: Optional[str] = None
    exp_name: str = f"train_hybrid_on_multicomp"
    from_pretrained: str = None

    param_transform: callable = SigmoidTransform

    channel_model_kwargs: dict = field(
        default_factory=lambda: {
            "width_size": 64,
            "depth_size": 3,
            "activation": jax.nn.tanh,
            "latent_states": ["z0", "z1", "z2", "z3", "z4"],
            "share_weights": True,
            "split_model": False,
            "restrict_state": False,
            "last_layer_initializer": jax.nn.initializers.constant(0.0), # normal_initializer(0.0, 0.01)
            "use_layer_norm": True,
        }
    )

    hh_model: HH = HH([Pospischil_Na(), Pospischil_K(), Pospischil_Leak(), Pospischil_Km()], [StepCurrent(amp=650.0, start=10.0, end=50.0)])  # no stimulus in soma compartment

    u_init: dict = field(
        default_factory=lambda: {
            "v": jnp.array(-70.0),
            "h": jnp.array(0.99968355),
            "m": jnp.array(0.00167569),
            "n": jnp.array(0.00654014),
            "p": jnp.array(4.63702142e-05),
            **{f"z{i}": 0.0*jr.normal(jr.PRNGKey(i+42)) for i in range(5)},
        }
    )

    integrate_kwargs: dict = field(
        default_factory=lambda: {
            "max_steps": 500_000,
            "rtol": 1e-6,
            "atol": 1e-8,
            "solver": diffrax.Tsit5(),
            "adjoint": diffrax.RecursiveCheckpointAdjoint(),
        }
    )

    trainable_params: list = field(
        default_factory=lambda: [
            "net",
            # ".initial_state['z0']",
            # ".initial_state['z1']",
            # ".initial_state['z2']",
            # ".initial_state['z3']",
            # ".initial_state['z4']",
            # ".initial_state['z5']",
            # ".initial_state['z6']",
            # ".initial_state['z7']",
            # ".initial_state['z8']",
            # ".initial_state['z9']",
            ".externals['i_ext'].amp",
        ]
    )

    init_params: dict = field(
        default_factory=lambda: {
            ".channels['pospischil_k'].ek": jnp.array(-90.0),
            ".channels['pospischil_k'].gk": jnp.array(5.0),
            ".channels['pospischil_k'].vt": jnp.array(-60.0),
            ".channels['pospischil_km'].ekm": jnp.array(-90.0),
            ".channels['pospischil_km'].gkm": jnp.array(0.004),
            ".channels['pospischil_km'].taumax": jnp.array(4000),
            ".channels['pospischil_leak'].el": jnp.array(-70.0),
            ".channels['pospischil_leak'].gl": jnp.array(0.1),
            ".channels['pospischil_na'].ena": jnp.array(50.0),
            ".channels['pospischil_na'].gna": jnp.array(50.0),
            ".channels['pospischil_na'].vt": jnp.array(-60.0),
            ".externals['i_ext'].amp": jnp.array(650.0 * 0.04), # ipt scaled by A_soma / A_total
            ".externals['i_ext'].end": jnp.array(50.0),
            ".externals['i_ext'].start": jnp.array(10.0),
            '.c': jnp.array(1.0),
            '.l': jnp.array(20.65533257),
            '.r': jnp.array(10.32766628),
            ".initial_state['v']": jnp.array(-70.0),
            ".initial_state['h']": jnp.array(0.99968355),
            ".initial_state['m']": jnp.array(0.00167569),
            ".initial_state['n']": jnp.array(0.00654014),
            ".initial_state['p']": jnp.array(4.63702142e-05),
        }
    )

    param_bounds: dict = field(
        default_factory=lambda: {
            # ".c": (0.5, 5.5),
            # ".r": (1.0, 25.0),
            # ".l": (5.0, 50.0),
            ".initial_state['z0']": (-1.0, 1.0),
            ".initial_state['z1']": (-1.0, 1.0),
            ".initial_state['z2']": (-1.0, 1.0),
            ".initial_state['z3']": (-1.0, 1.0),
            ".initial_state['z4']": (-1.0, 1.0),
            ".initial_state['z5']": (-1.0, 1.0),
            ".initial_state['z6']": (-1.0, 1.0),
            ".initial_state['z7']": (-1.0, 1.0),
            ".initial_state['z8']": (-1.0, 1.0),
            ".initial_state['z9']": (-1.0, 1.0),
            # ".channels['pospischil_k'].ek": (-100.0, -40.0),
            # ".channels['pospischil_k'].gk": (1.0, 20.0),
            # ".channels['pospischil_k'].vt": (-80.0, -40.0),
            # ".channels['pospischil_km'].ekm": (-100.0, -40.0),
            # ".channels['pospischil_km'].gkm": (1e-4, 1e-1),
            # ".channels['pospischil_km'].taumax": (1e3, 8e3),
            # ".channels['pospischil_leak'].el": (-80.0, -60.0),
            # ".channels['pospischil_leak'].gl": (0.01, 0.5),
            # ".channels['pospischil_na'].ena": (30.0, 60.0),
            # ".channels['pospischil_na'].gna": (10.0, 100.0),
            # ".channels['pospischil_na'].vt": (-80.0, -40.0),
            ".externals['i_ext'].amp": (0.0, 650.0),
            # ".externals['i_ext'].start": (0.0, 20.0),
            # ".externals['i_ext'].end": (20.0, 80.0),
        }
    )

    opt_strategy: list[dict] = field(
        default_factory=lambda: [
            {
                "ode_optim": optax.adam(
                    learning_rate=optax.cosine_onecycle_schedule(200, 1e-1),
                    # learning_rate=optax.cosine_decay_schedule(1e-1, 300),
                ),
                "nn_optim": optax.adamw(
                    # learning_rate=optax.cosine_decay_schedule(1e-3, 300),
                    learning_rate=optax.cosine_onecycle_schedule(200, 1e-4),
                    weight_decay=1e-5,
                ),
                "max_steps": 200,
                "length_frac": 0.22,
                "mask_grads": "nn",
            },
            {
                "ode_optim": optax.adam(
                    learning_rate=optax.cosine_onecycle_schedule(300, 1e-1),
                    # learning_rate=optax.cosine_decay_schedule(1e-1, 300),
                ),
                "nn_optim": optax.adamw(
                    # learning_rate=optax.cosine_decay_schedule(1e-3, 300),
                    learning_rate=optax.cosine_onecycle_schedule(300, 1e-4),
                    weight_decay=1e-5,
                ),
                "max_steps": 300,
                "length_frac": 0.28,
                "mask_grads": "nn",
            },
            {
                "ode_optim": optax.adam(
                    learning_rate=optax.cosine_onecycle_schedule(500, 1e-1),
                    # learning_rate=optax.cosine_decay_schedule(1e-1, 300),
                ),
                "nn_optim": optax.adamw(
                    # learning_rate=optax.cosine_decay_schedule(1e-3, 300),
                    learning_rate=optax.cosine_onecycle_schedule(500, 1e-3),
                    weight_decay=1e-5,
                ),
                "max_steps": 500,
                "length_frac": 0.28,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(
                    # learning_rate=optax.cosine_decay_schedule(1e-2, 500),
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-1),
                ),
                "nn_optim": optax.adamw(
                    # learning_rate=optax.cosine_decay_schedule(1e-3, 500),
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-3),
                    weight_decay=1e-5,
                ),
                "max_steps": 1000,
                "length_frac": 0.39,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-1),
                    # learning_rate=optax.cosine_decay_schedule(1e-1, 500),
                ),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-5),
                    # learning_rate=optax.cosine_decay_schedule(1e-3, 500),
                    weight_decay=1e-5,
                ),
                "max_steps": 1000,
                "length_frac": 1.0,
                "mask_grads": None,
            },
        ]
    )


@dataclass
class train_node_on_multicomp:
    output_dir: str = "."
    seed: int = 0
    batch_size: int = 1
    ChannelModel: Channel = NODE  # works best with mish activation and plain Adam
    train_fpath: str = "data/hh_synth/pospischil_mutlicomp_single_pt_soma.json"
    obs_noise: float = 0.0
    val_fpath: Optional[str] = None
    exp_name: str = f"train_node_on_multicomp"
    from_pretrained: str = None

    param_transform: callable = SigmoidTransform

    channel_model_kwargs: dict = field(
        default_factory=lambda: {
            "width_size": 64,
            "depth_size": 3,
            "activation": jax.nn.tanh,
            "latent_states": ["z0", "z1", "z2", "z3", "z4"],
            "share_weights": True,
            "split_model": False,
            "restrict_state": False,
            "last_layer_initializer": jax.nn.initializers.constant(0.0), # normal_initializer(0.0, 0.01)
            "use_layer_norm": True,
            "return_current": False,
        }
    )

    hh_model: HH = HH([Pospischil_Na(), Pospischil_K(), Pospischil_Leak(), Pospischil_Km()], [StepCurrent(amp=650.0, start=10.0, end=50.0)])  # no stimulus in soma compartment

    u_init: dict = field(
        default_factory=lambda: {
            "v": jnp.array(-70.0),
            **{f"z{i}": 0.0*jr.normal(jr.PRNGKey(i+42)) for i in range(10)},
        }
    )

    integrate_kwargs: dict = field(
        default_factory=lambda: {
            "max_steps": 500_000,
            "rtol": 1e-6,
            "atol": 1e-8,
            "solver": diffrax.Tsit5(),
            "adjoint": diffrax.RecursiveCheckpointAdjoint(),
        }
    )

    trainable_params: list = field(
        default_factory=lambda: [
            "net",
            # NOTE: currently inital values not working
            # ".initial_state['v']",
            # ".initial_state['z0']",
            # ".initial_state['z1']",
            # ".initial_state['z2']",
            # ".initial_state['z3']",
            # ".initial_state['z4']",
        ]
    )

    opt_strategy: list[dict] = field(
        default_factory=lambda: [
            {
                "ode_optim": optax.adam(
                    learning_rate=optax.cosine_onecycle_schedule(200, 1e-1),
                    # learning_rate=optax.cosine_decay_schedule(1e-1, 300),
                ),
                "nn_optim": optax.adamw(
                    # learning_rate=optax.cosine_decay_schedule(1e-3, 300),
                    learning_rate=optax.cosine_onecycle_schedule(200, 1e-4),
                    weight_decay=1e-5,
                ),
                "max_steps": 200,
                "length_frac": 0.22,
                "mask_grads": "nn",
            },
            {
                "ode_optim": optax.adam(
                    learning_rate=optax.cosine_onecycle_schedule(300, 1e-1),
                    # learning_rate=optax.cosine_decay_schedule(1e-1, 300),
                ),
                "nn_optim": optax.adamw(
                    # learning_rate=optax.cosine_decay_schedule(1e-3, 300),
                    learning_rate=optax.cosine_onecycle_schedule(300, 1e-4),
                    weight_decay=1e-5,
                ),
                "max_steps": 300,
                "length_frac": 0.28,
                "mask_grads": "nn",
            },
            {
                "ode_optim": optax.adam(
                    learning_rate=optax.cosine_onecycle_schedule(500, 1e-1),
                    # learning_rate=optax.cosine_decay_schedule(1e-1, 300),
                ),
                "nn_optim": optax.adamw(
                    # learning_rate=optax.cosine_decay_schedule(1e-3, 300),
                    learning_rate=optax.cosine_onecycle_schedule(500, 1e-3),
                    weight_decay=1e-5,
                ),
                "max_steps": 500,
                "length_frac": 0.28,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(
                    # learning_rate=optax.cosine_decay_schedule(1e-2, 500),
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-1),
                ),
                "nn_optim": optax.adamw(
                    # learning_rate=optax.cosine_decay_schedule(1e-3, 500),
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-3),
                    weight_decay=1e-5,
                ),
                "max_steps": 1000,
                "length_frac": 0.39,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-1),
                    # learning_rate=optax.cosine_decay_schedule(1e-1, 500),
                ),
                "nn_optim": optax.adamw(
                    learning_rate=optax.cosine_onecycle_schedule(1000, 1e-5),
                    # learning_rate=optax.cosine_decay_schedule(1e-3, 500),
                    weight_decay=1e-5,
                ),
                "max_steps": 1000,
                "length_frac": 1.0,
                "mask_grads": None,
            },
        ]
    )


@dataclass
class train_soma_on_multicomp(train_hybrid_on_multicomp):
    output_dir: str = "."
    seed: int = 0
    batch_size: int = 1
    ChannelModel: Channel = NODE  # works best with mish activation and plain Adam
    train_fpath: str = "data/hh_synth/pospischil_mutlicomp_single_pt_soma.json"
    obs_noise: float = 0.0
    param_noise: float = 0.0
    val_fpath: Optional[str] = None
    exp_name: str = f"train_soma_on_multicomp"
    from_pretrained: str = None

    param_transform: callable = SigmoidTransform

    hh_model: HH = HH([Pospischil_Na(), Pospischil_K(), Pospischil_Leak(), Pospischil_Km()], [StepCurrent(amp=650.0, start=10.0, end=50.0)])  # no stimulus in soma compartment

    u_init: dict = field(
        default_factory=lambda: {
            "v": jnp.array(-70.0),
            "h": jnp.array(0.99968355),
            "m": jnp.array(0.00167569),
            "n": jnp.array(0.00654014),
            "p": jnp.array(4.63702142e-05),
        }
    )

    integrate_kwargs: dict = field(
        default_factory=lambda: {
            "max_steps": 500_000,
            "rtol": 1e-6,
            "atol": 1e-8,
            "solver": diffrax.Tsit5(),
            "adjoint": diffrax.RecursiveCheckpointAdjoint(),
        }
    )

    trainable_params: list = field(
        default_factory=lambda: [
            '.l',
            '.r',
            '.c',
            # ".externals['i_ext'].amp",
            
            # ".channels['pospischil_k'].ek",
            # ".channels['pospischil_k'].gk",
            # ".channels['pospischil_k'].vt",
            # ".channels['pospischil_km'].ekm",
            # ".channels['pospischil_km'].gkm",
            # ".channels['pospischil_km'].taumax",
            # ".channels['pospischil_leak'].el",
            # ".channels['pospischil_leak'].gl",
            # ".channels['pospischil_na'].ena",
            # ".channels['pospischil_na'].gna",
            # ".channels['pospischil_na'].vt",

            # ".initial_state['v']",
            # ".initial_state['h']",
            # ".initial_state['m']",
            # ".initial_state['n']",
            # ".initial_state['p']",
        ]
    )

    init_params: dict = field(
        default_factory=lambda: {
            ".channels['pospischil_k'].ek": jnp.array(-90.0),
            ".channels['pospischil_k'].gk": jnp.array(5.0),
            ".channels['pospischil_k'].vt": jnp.array(-60.0),
            ".channels['pospischil_km'].ekm": jnp.array(-90.0),
            ".channels['pospischil_km'].gkm": jnp.array(0.004),
            ".channels['pospischil_km'].taumax": jnp.array(4000),
            ".channels['pospischil_leak'].el": jnp.array(-70.0),
            ".channels['pospischil_leak'].gl": jnp.array(0.1),
            ".channels['pospischil_na'].ena": jnp.array(50.0),
            ".channels['pospischil_na'].gna": jnp.array(50.0),
            ".channels['pospischil_na'].vt": jnp.array(-60.0),
            ".externals['i_ext'].amp": jnp.array(650.0),
            ".externals['i_ext'].end": jnp.array(50.0),
            ".externals['i_ext'].start": jnp.array(10.0),
            '.c': jnp.array(1.0),
            '.l': jnp.array(20.65533257),
            '.r': jnp.array(10.32766628 * 10), # scale up to approx multicompartment
            # ".initial_state['v']": jnp.array(-70.0),
            # ".initial_state['h']": jnp.array(0.99968355),
            # ".initial_state['m']": jnp.array(0.00167569),
            # ".initial_state['n']": jnp.array(0.00654014),
            # ".initial_state['p']": jnp.array(4.63702142e-05),
        }
    )

    param_bounds: dict = field(
        default_factory=lambda: {
            ".c": (0.5, 5.5),
            ".r": (1.0, 500.0),
            ".l": (5.0, 100.0),
            ".channels['pospischil_k'].ek": (-100.0, -40.0),
            ".channels['pospischil_k'].gk": (1.0, 20.0),
            ".channels['pospischil_k'].vt": (-80.0, -40.0),
            ".channels['pospischil_km'].ekm": (-100.0, -40.0),
            ".channels['pospischil_km'].gkm": (1e-4, 1e-1),
            ".channels['pospischil_km'].taumax": (1e3, 8e3),
            ".channels['pospischil_leak'].el": (-80.0, -60.0),
            ".channels['pospischil_leak'].gl": (0.01, 0.5),
            ".channels['pospischil_na'].ena": (30.0, 60.0),
            ".channels['pospischil_na'].gna": (10.0, 100.0),
            ".channels['pospischil_na'].vt": (-80.0, -40.0),
            ".externals['i_ext'].amp": (0.0, 650.0),
        }
    )

    opt_strategy: list[dict] = field(
        default_factory=lambda: [
            {
                "ode_optim": optax.adam(
                    learning_rate=optax.cosine_onecycle_schedule(200, 1e-3),
                ),
                "max_steps": 200,
                "length_frac": 0.28,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(
                    learning_rate=optax.cosine_onecycle_schedule(300, 1e-2),
                ),
                "max_steps": 300,
                "length_frac": 0.39,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(
                    learning_rate=optax.cosine_onecycle_schedule(500, 1e-2),
                ),
                "max_steps": 500,
                "length_frac": 1.0,
                "mask_grads": None,
            },
        ]
    )

@dataclass
class train_omni_on_hh_k(train_hybrid_on_hh):
    ChannelModel: Channel = Omni
    exp_name: str = f"train_omni_on_hh_k"

    channel_model_kwargs: dict = field(
        default_factory=lambda: {"latent_states": ["n"]}
    )

    integrate_kwargs: dict = field(
        default_factory=lambda: {
            "max_steps": 50_000,
            "rtol": 1e-6,
            "atol": 1e-8,
            "solver": diffrax.Tsit5(),
            "adjoint": diffrax.RecursiveCheckpointAdjoint(),
        }
    )

    init_params: dict = field(
        default_factory=lambda: {
            ".c": jnp.array(1.0),
            ".tadj": jnp.array(3.0 ** ((37.0 - 37.0) / 10.0)),
            ".r": jnp.array(1.0),
            ".l": jnp.array(10.0),
            ".channels['leak'].gl": jnp.array(0.3),
            ".channels['leak'].el": jnp.array(-70.0),
            ".channels['leak'].tadj": jnp.array(3.0 ** ((37.0 - 37.0) / 10.0)),
            ".channels['na'].gna": jnp.array(120.0),
            ".channels['na'].ena": jnp.array(50.0),
            ".channels['na'].tadj": jnp.array(3.0 ** ((37.0 - 37.0) / 10.0)),
            ".channels['k'].gk": jnp.array(36.0),
            ".channels['k'].ek": jnp.array(-75.0),
            ".channels['k'].tadj": jnp.array(3.0 ** ((37.0 - 37.0) / 10.0)),
            
            ".channels['omni'].gates['m'].vh": jnp.array(-30.0),
            ".channels['omni'].gates['m'].A": jnp.array(10.0),
            ".channels['omni'].gates['m'].a": jnp.array(0.10),
            ".channels['omni'].gates['m'].b": jnp.array(-5.0),
            ".channels['omni'].gates['m'].b1": jnp.array(-1e-2),
            ".channels['omni'].gates['m'].c1": jnp.array(1e-3),
            ".channels['omni'].gates['m'].d1": jnp.array(0.0),
            ".channels['omni'].gates['m'].e1": jnp.array(0.0),
            ".channels['omni'].gates['m'].b2": jnp.array(0.0),
            ".channels['omni'].gates['m'].c2": jnp.array(-1e-3),
            ".channels['omni'].gates['m'].d2": jnp.array(0.0),
            ".channels['omni'].gates['m'].e2": jnp.array(0.0),
            
            ".channels['omni'].gates['h'].vh": jnp.array(-60.0),
            ".channels['omni'].gates['h'].A": jnp.array(3000.0),
            ".channels['omni'].gates['h'].a": jnp.array(-0.2),
            ".channels['omni'].gates['h'].b": jnp.array(10.0),
            ".channels['omni'].gates['h'].b1": jnp.array(-1e-1),
            ".channels['omni'].gates['h'].c1": jnp.array(1e-3),
            ".channels['omni'].gates['h'].d1": jnp.array(1e-3),
            ".channels['omni'].gates['h'].e1": jnp.array(0.0),
            ".channels['omni'].gates['h'].b2": jnp.array(0.0),
            ".channels['omni'].gates['h'].c2": jnp.array(-1e-3),
            ".channels['omni'].gates['h'].d2": jnp.array(0.0),
            ".channels['omni'].gates['h'].e2": jnp.array(0.0),
            
            ".channels['omni'].gates['n'].vh": jnp.array(-30.0),
            ".channels['omni'].gates['n'].A": jnp.array(1.0),
            ".channels['omni'].gates['n'].a": jnp.array(-0.1),
            ".channels['omni'].gates['n'].b": jnp.array(-2.0),
            ".channels['omni'].gates['n'].b1": jnp.array(-1e-2),
            ".channels['omni'].gates['n'].c1": jnp.array(1e-4),
            ".channels['omni'].gates['n'].d1": jnp.array(0.0),
            ".channels['omni'].gates['n'].e1": jnp.array(0.0),
            ".channels['omni'].gates['n'].b2": jnp.array(0.01),
            ".channels['omni'].gates['n'].c2": jnp.array(1e-5),
            ".channels['omni'].gates['n'].d2": jnp.array(0.0),
            ".channels['omni'].gates['n'].e2": jnp.array(0.0),
            
            ".channels['node'].gx": jnp.array(36.0),
            ".channels['node'].ex": jnp.array(-75.0),
            
            ".channels['omni'].powx['m']": jnp.array(3.0),
            ".channels['omni'].powx['h']": jnp.array(1.0),
            ".channels['omni'].powx['n']": jnp.array(4.0),
        }
    )

    trainable_params: list = field(
        default_factory=lambda: [
            # ".c",  # Membrane capacitance (uF/cm^2)
            # ".tadj",  # Temperature adjustment factor (Q10 scaling)
            # ".r",  # radius
            # ".l",  # length
            # ".channels['leak'].gl",  # Leak conductance (mS/cm^2)
            # ".channels['leak'].el",  # Leak reversal potential (mV)
            # ".channels['leak'].tadj",  # Leak temperature adjustment (Q10 scaling)
            # ".channels['na'].gna",  # Sodium conductance (mS/cm^2)
            # ".channels['na'].ena",  # Sodium reversal potential (mV)
            # ".channels['na'].tadj",  # Sodium temperature adjustment (Q10 scaling)
            # ".channels['k'].gk",    # Potassium conductance (mS/cm^2)
            # ".channels['k'].ek",    # Potassium reversal potential (mV)
            # ".channels['k'].tadj",  # Potassium temperature adjustment (Q10 scaling)

            # m-gate (activation) parameters for Omnichannel
            ".channels['omni'].gates['m'].vh",  # Half-activation voltage (mV)
            ".channels['omni'].gates['m'].A",  # Time constant amplitude (ms)
            ".channels['omni'].gates['m'].a",  # Slope parameter for u_inf
            ".channels['omni'].gates['m'].b",  # Shift parameter for u_inf
            ".channels['omni'].gates['m'].c",  # Maximum value for u_inf
            ".channels['omni'].gates['m'].d",  # Minimum value for u_inf
            # m-gate B(v) and C(v) polynomial coefficients
            ".channels['omni'].gates['m'].b1",  # B(v) 1st order coefficient
            ".channels['omni'].gates['m'].c1",  # B(v) 2nd order coefficient
            ".channels['omni'].gates['m'].d1",  # B(v) 3rd order coefficient
            ".channels['omni'].gates['m'].e1",  # B(v) 4th order coefficient
            ".channels['omni'].gates['m'].b2",  # C(v) 1st order coefficient
            ".channels['omni'].gates['m'].c2",  # C(v) 2nd order coefficient
            ".channels['omni'].gates['m'].d2",  # C(v) 3rd order coefficient
            ".channels['omni'].gates['m'].e2",  # C(v) 4th order coefficient
            # h-gate (inactivation) parameters for Omnichannel
            ".channels['omni'].gates['h'].vh",  # Half-activation voltage (mV)
            ".channels['omni'].gates['h'].A",  # Time constant amplitude (ms)
            ".channels['omni'].gates['h'].a",  # Slope parameter for u_inf
            ".channels['omni'].gates['h'].b",  # Shift parameter for u_inf
            ".channels['omni'].gates['h'].c",  # Maximum value for u_inf
            ".channels['omni'].gates['h'].d",  # Minimum value for u_inf
            # h-gate B(v) and C(v) polynomial coefficients
            ".channels['omni'].gates['h'].b1",  # B(v) 1st order coefficient
            ".channels['omni'].gates['h'].c1",  # B(v) 2nd order coefficient
            ".channels['omni'].gates['h'].d1",  # B(v) 3rd order coefficient
            ".channels['omni'].gates['h'].e1",  # B(v) 4th order coefficient
            ".channels['omni'].gates['h'].b2",  # C(v) 1st order coefficient
            ".channels['omni'].gates['h'].c2",  # C(v) 2nd order coefficient
            ".channels['omni'].gates['h'].d2",  # C(v) 3rd order coefficient
            ".channels['omni'].gates['h'].e2",  # C(v) 4th order coefficient
            # n-gate (K+ channel activation) parameters for Omnichannel
            ".channels['omni'].gates['n'].vh",  # Half-activation voltage (mV)
            ".channels['omni'].gates['n'].A",  # Time constant amplitude (ms)
            ".channels['omni'].gates['n'].a",  # Slope parameter for u_inf
            ".channels['omni'].gates['n'].b",  # Shift parameter for u_inf
            ".channels['omni'].gates['n'].c",  # Maximum value for u_inf
            ".channels['omni'].gates['n'].d",  # Minimum value for u_inf
            # n-gate B(v) and C(v) polynomial coefficients
            ".channels['omni'].gates['n'].b1",  # B(v) 1st order coefficient
            ".channels['omni'].gates['n'].c1",  # B(v) 2nd order coefficient
            # ".channels['omni'].gates['n'].d1",  # B(v) 3rd order coefficient
            # ".channels['omni'].gates['n'].e1",  # B(v) 4th order coefficient
            ".channels['omni'].gates['n'].b2",  # C(v) 1st order coefficient
            ".channels['omni'].gates['n'].c2",  # C(v) 2nd order coefficient
            # ".channels['omni'].gates['n'].d2",  # C(v) 3rd order coefficient
            # ".channels['omni'].gates['n'].e2",  # C(v) 4th order coefficient
            
            # # Omnichannel conductance, reversal potential, and exponents
            # ".channels['omni'].gx",  # Channel conductance (mS/cm^2)
            # ".channels['omni'].ex",  # Channel reversal potential (mV)
            # ".channels['omni'].powx['m']",  # Power for m gating variable
            # ".channels['omni'].powx['h']",  # Power for h gating variable
            # ".channels['omni'].powx['n']",  # Power for n gating variable
        ]
    )

    param_bounds: dict = field(
        default_factory=lambda: {
            ".c": (0.5, 1.5),
            ".tadj": (1.0 ** ((34.0 - 37.0) / 10.0), 1.5 ** ((37.0 - 34.0) / 10.0)),
            ".r": (0.5, 1.5),
            ".l": (1.0, 20.0),
            ".channels['k'].gk": (20.0, 40.0),
            ".channels['k'].ek": (-90.0, -50.0),
            ".channels['k'].tadj": (
                2.5 ** ((34.0 - 37.0) / 10.0),
                3.5 ** ((37.0 - 34.0) / 10.0),
            ),
            ".channels['leak'].gl": (0.01, 0.5),
            ".channels['leak'].el": (-80.0, -60.0),
            ".channels['leak'].tadj": (
                2.5 ** ((34.0 - 37.0) / 10.0),
                3.5 ** ((37.0 - 34.0) / 10.0),
            ),
            ".channels['na'].gna": (100.0, 200.0),
            ".channels['na'].ena": (30.0, 60.0),
            ".channels['na'].tadj": (
                2.5 ** ((34.0 - 37.0) / 10.0),
                3.5 ** ((37.0 - 34.0) / 10.0),
            ),

            ".channels['omni'].gates['m'].vh": (-500, 500),
            ".channels['omni'].gates['m'].A": (1e-3, 1e4),
            ".channels['omni'].gates['m'].a": (-1.0, 1.0),
            ".channels['omni'].gates['m'].b": (-1e2, 2e1),
            ".channels['omni'].gates['m'].b1": (-1e2, 1e2),
            ".channels['omni'].gates['m'].c1": (-1e2, 1e2),
            ".channels['omni'].gates['m'].d1": (-1e2, 1e2),
            ".channels['omni'].gates['m'].e1": (-1e2, 1e2),
            ".channels['omni'].gates['m'].b2": (-1e2, 1e2),
            ".channels['omni'].gates['m'].c2": (-1e2, 1e2),
            ".channels['omni'].gates['m'].d2": (-1e2, 1e2),
            ".channels['omni'].gates['m'].e2": (-1e2, 1e2),
            
            ".channels['omni'].gates['h'].vh": (-500, 500),
            ".channels['omni'].gates['h'].A": (1e-3, 1e4),
            ".channels['omni'].gates['h'].a": (-1.0, 1.0),
            ".channels['omni'].gates['h'].b": (-1e2, 2e1),
            ".channels['omni'].gates['h'].b1": (-1e2, 1e2),
            ".channels['omni'].gates['h'].c1": (-1e2, 1e2),
            ".channels['omni'].gates['h'].d1": (-1e2, 1e2),
            ".channels['omni'].gates['h'].e1": (-1e2, 1e2),
            ".channels['omni'].gates['h'].b2": (-1e2, 1e2),
            ".channels['omni'].gates['h'].c2": (-1e2, 1e2),
            ".channels['omni'].gates['h'].d2": (-1e2, 1e2),
            ".channels['omni'].gates['h'].e2": (-1e2, 1e2),
            
            ".channels['omni'].gates['n'].vh": (-500, 500),
            ".channels['omni'].gates['n'].A": (1e-3, 1e4),
            ".channels['omni'].gates['n'].a": (-1.0, 1.0),
            ".channels['omni'].gates['n'].b": (-1e2, 2e1),
            ".channels['omni'].gates['n'].b1": (-1e2, 1e2),
            ".channels['omni'].gates['n'].c1": (-1e2, 1e2),
            ".channels['omni'].gates['n'].d1": (-1e2, 1e2),
            ".channels['omni'].gates['n'].e1": (-1e2, 1e2),
            ".channels['omni'].gates['n'].b2": (-1e2, 1e2),
            ".channels['omni'].gates['n'].c2": (-1e2, 1e2),
            ".channels['omni'].gates['n'].d2": (-1e2, 1e2),
            ".channels['omni'].gates['n'].e2": (-1e2, 1e2),
            
            ".channels['omni'].gx": (10.0, 200.0),
            ".channels['omni'].ex": (-100.0, 100.0),
            
            ".channels['omni'].powx['m']": (2.5, 3.5),
            ".channels['omni'].powx['h']": (0.5, 1.5),
            ".channels['omni'].powx['n']": (3.5, 5.5),
        }
    )

    opt_strategy: list[dict] = field(
        default_factory=lambda: [
            {
                "ode_optim": optax.adam(learning_rate=optax.warmup_constant_schedule(1e-3, 1e-1, 100)),
                "nn_optim": optax.adam(learning_rate=optax.warmup_constant_schedule(1e-3, 1e-1, 100)),
                "max_steps": 500,
                "length_frac": 0.22,
                "mask_grads": None,
            },
            {
                "ode_optim": optax.adam(learning_rate=optax.warmup_constant_schedule(1e-3, 1e-1, 200)),
                "nn_optim": optax.adam(learning_rate=optax.warmup_constant_schedule(1e-3, 1e-1, 200)),
                "max_steps": 1000,
                "length_frac": 1.0,
                "mask_grads": None,
            },
        ]
    )