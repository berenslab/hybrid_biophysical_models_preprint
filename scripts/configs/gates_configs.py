from dataclasses import dataclass, field
from typing import Optional

from hybrid_models.utils import absorb_kwargs
from hybrid_models.hh.channels import *
from hybrid_models.hh.base import ExternalInput, StepCurrent
from hybrid_models.transforms import (
    LogisticTransform,
    IdentityTransform,
    SigmoidTransform,
)
from hybrid_models.optimizers import preconditioned_sgld, sgld

from hybrid_models.hh import HH
import optax
import jax
import jax.numpy as jnp


@dataclass
class generate_hh_channel_data:
    output_dir: str = "data/hh_channels"
    seed: int = 0
    observed_states: str = "m, h"
    fname: str = f"na_single"
    channel_model: Channel = Na()

    train_noise_std: float = 0.0
    val_noise_std: float = 0.0

    v_train = jnp.linspace(-100.0, 100.0, 50)
    v_val = jnp.linspace(-100.0, 100.0, 50)


@dataclass
class train_node_on_gates:
    output_dir: str = "."
    seed: int = 0
    batch_size: int = -1
    train_fpath: str = "data/icg_channels/icg-channels-Na/101629_na3n.json"
    exp_name: str = f"train_icg_gate"
    val_fpath: Optional[str] = None

    param_transform: callable = IdentityTransform  # dont use transform for this

    ChannelModel: Channel = BioPhysicsNODE1
    channel_model_kwargs: dict = field(
        default_factory=lambda: {
            "width_size": 32,
            "depth_size": 2,
            "activation": jax.nn.tanh,
            "use_layer_norm": True,
            # "last_layer_initializer": jax.nn.initializers.normal(stddev=0.1),
        }
    )
    init_params: dict = field(default_factory=lambda: {})

    trainable_params: list = field(
        default_factory=lambda: [
            "net",
        ]
    )

    param_bounds: dict = field(default_factory=lambda: {})

    opt_strategy: list[dict] = field(
        default_factory=lambda: [
            {
                "optimizer": optax.adamw(
                    # learning_rate=optax.cosine_onecycle_schedule(10_000, 1e-3),
                    learning_rate=optax.warmup_constant_schedule(1e-6, 1e-3, 1000),
                    weight_decay=1e-4,
                ),
                "max_steps": 10_000,
            },
            {
                "optimizer": optax.adamw(
                    # learning_rate=optax.cosine_onecycle_schedule(10_000, 1e-3),
                    learning_rate=optax.warmup_cosine_decay_schedule(1e-6, 1e-3, 3000, 7000),
                    weight_decay=1e-4,
                ),
                "max_steps": 10_000,
            },
        ]
    )


@dataclass
class train_omni_on_gates:
    output_dir: str = "."
    seed: int = 0
    batch_size: int = -1
    exp_name: str = f"train_k_gate"
    train_fpath: str = "data/icg_channels/icg-channels-K/279_hh2.json"
    val_fpath: Optional[str] = None

    param_transform: callable = LogisticTransform

    ChannelModel: Channel = Omni
    channel_model_kwargs: dict = field(default_factory=lambda: {})

    init_params: dict = field(default_factory=lambda: {})

    trainable_params: list = field(
        default_factory=lambda: [
            ".gates['m'].vh",  # Half-activation voltage (mV)
            ".gates['m'].A",  # Time constant amplitude (ms)
            ".gates['m'].a",  # Slope parameter for u_inf
            ".gates['m'].b",  # Shift parameter for u_inf
            # ".gates['m'].c",  # Maximum value for u_inf
            # ".gates['m'].d",  # Minimum value for u_inf
            ".gates['m'].b1",  # B(v) polynomial coefficients
            ".gates['m'].c1",  # B(v) polynomial coefficients
            ".gates['m'].d1",  # B(v) polynomial coefficients
            ".gates['m'].e1",  # B(v) polynomial coefficients
            ".gates['m'].b2",  # C(v) polynomial coefficients
            ".gates['m'].c2",  # C(v) polynomial coefficients
            ".gates['m'].d2",  # C(v) polynomial coefficients
            ".gates['m'].e2",  # C(v) polynomial coefficients
            ".gates['h'].vh",  # Half-activation voltage (mV)
            ".gates['h'].A",  # Time constant amplitude (ms)
            ".gates['h'].a",  # Slope parameter for u_inf
            ".gates['h'].b",  # Shift parameter for u_inf
            # ".gates['h'].c",  # Maximum value for u_inf
            # ".gates['h'].d",  # Minimum value for u_inf
            ".gates['h'].b1",  # B(v) polynomial coefficient
            ".gates['h'].c1",  # B(v) polynomial coefficient
            ".gates['h'].d1",  # B(v) polynomial coefficient
            ".gates['h'].e1",  # B(v) polynomial coefficient
            ".gates['h'].b2",  # C(v) polynomial coefficient
            ".gates['h'].c2",  # C(v) polynomial coefficient
            ".gates['h'].d2",  # C(v) polynomial coefficient
            ".gates['h'].e2",  # C(v) polynomial coefficient
            ".gates['n'].vh",  # Half-activation voltage (mV)
            ".gates['n'].A",  # Time constant amplitude (ms)
            ".gates['n'].a",  # Slope parameter for u_inf
            ".gates['n'].b",  # Shift parameter for u_inf
            # ".gates['n'].c",  # Maximum value for u_inf
            # ".gates['n'].d",  # Minimum value for u_inf
            ".gates['n'].b1",  # B(v) polynomial coefficient
            ".gates['n'].c1",  # B(v) polynomial coefficient
            ".gates['n'].d1",  # B(v) polynomial coefficient
            ".gates['n'].e1",  # B(v) polynomial coefficient
            ".gates['n'].b2",  # C(v) polynomial coefficient
            ".gates['n'].c2",  # C(v) polynomial coefficient
            ".gates['n'].d2",  # C(v) polynomial coefficient
            ".gates['n'].e2",  # C(v) polynomial coefficient
        ]
    )

    param_bounds: dict = field(
        default_factory=lambda: {
            ".gates['m'].vh": (-5e2, 3e2),
            ".gates['m'].A": (1e-2, 3e3),
            ".gates['m'].a": (8e-2, 3e-1),
            ".gates['m'].b": (-2e1, -2e-1),
            # ".gates['m'].c": (9.9e-1, 1.01),
            # ".gates['m'].d": (-1e-2, 1e-2),
            ".gates['m'].b1": (-1e2, 1.0),
            ".gates['m'].c1": (-2e-2, 5e-1),
            ".gates['m'].d1": (-2e-3, 3e-3),
            ".gates['m'].e1": (-6e-4, 3e-3),
            ".gates['m'].b2": (-1e-4, 1e-4),
            ".gates['m'].c2": (-6e-1, 5e-1),
            ".gates['m'].d2": (-6e-2, 1e-2),
            ".gates['m'].e2": (-1e-4, 1e-4),
            ".gates['h'].vh": (-7e2, 1e2),
            ".gates['h'].A": (1e-1, 4e4),
            ".gates['h'].a": (-5e-1, -9e-2),
            ".gates['h'].b": (1.0, 3e1),
            # ".gates['h'].c": (9.9e-1, 1.01),
            # ".gates['h'].d": (-1e-2, 1e-2),
            ".gates['h'].b1": (-1e2, 3.0),
            ".gates['h'].c1": (-2e-2, 5e-1),
            ".gates['h'].d1": (-3e-4, 3e-3),
            ".gates['h'].e1": (-3e-3, 4e-4),
            ".gates['h'].b2": (-1e-4, 1e-4),
            ".gates['h'].c2": (-9e-1, 4e-1),
            ".gates['h'].d2": (-9e-2, 4e-3),
            ".gates['h'].e2": (-1e-4, 1e-4),
            ".gates['n'].vh": (-4e2, 4e2),
            ".gates['n'].A": (1e-1, 8e3),
            ".gates['n'].a": (-3e-1, 7e-1),
            ".gates['n'].b": (-1e2, 2e1),
            # ".gates['n'].c": (9.9e-1, 1.01),
            # ".gates['n'].d": (-1e-2, 1e-2),
            ".gates['n'].b1": (-3.2, 3e1),
            ".gates['n'].c1": (-1e-2, 1e-1),
            ".gates['n'].d1": (-2e-3, 2.0e-3),
            ".gates['n'].e1": (-3e-4, 3e1),
            ".gates['n'].b2": (-1e-4, 1e-4),
            ".gates['n'].c2": (-6e-1, 9.0),
            ".gates['n'].d2": (-7.0, 2e-2),
            ".gates['n'].e2": (-1e-4, 3e-1),
        }
    )

    opt_strategy: list[dict] = field(
        default_factory=lambda: [
            {
                "optimizer": optax.adam(
                    # learning_rate=optax.cosine_onecycle_schedule(10_000, 1e-3),
                    learning_rate=optax.warmup_constant_schedule(1e-6, 1e-3, 1000),
                ),
                "max_steps": 10_000,
            },
            {
                "optimizer": optax.adam(
                    # learning_rate=optax.cosine_onecycle_schedule(10_000, 1e-3),
                    learning_rate=optax.warmup_cosine_decay_schedule(1e-6, 1e-3, 3000, 7000),
                ),
                "max_steps": 10_000,
            },
        ]
    )
