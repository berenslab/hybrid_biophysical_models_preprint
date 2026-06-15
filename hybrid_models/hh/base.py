import jax.numpy as jnp
import diffrax
import functools

import equinox as eqx
from typing import Union, Optional
import jax

from .channels import Channel
from hybrid_models.utils import tree_set_with_path
import lineax


class ExternalInput(eqx.Module):
    name: str = eqx.field(static=True)

    def __init__(self, name):
        self.name = name

    def __call__(self, t, u, *args, **kwargs):
        return u

    def __repr__(self):
        cls_name = self.__class__.__name__
        fields = self.__dataclass_fields__.keys()
        params = {k: getattr(self, k) for k in fields}
        del params["name"]
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        return f"{cls_name}({params_str})"


class StepCurrent(ExternalInput):
    start: float
    end: float
    amp: float

    def __init__(self, name="i_ext", start=5.0, end=20.0, amp=5.0):
        super().__init__(name)
        self.start = start
        self.end = end
        self.amp = amp

    def __call__(self, t, u, *args, **kwargs):
        return self.amp * (t >= self.start) * (t <= self.end)


class HH(eqx.Module):
    channels: dict[str, Channel] = eqx.field(converter=dict)
    c: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)
    r: jnp.ndarray = eqx.field(converter=jnp.array)
    l: jnp.ndarray = eqx.field(converter=jnp.array)
    externals: dict[str, ExternalInput] = eqx.field(converter=dict)
    initial_state: dict[str, jnp.ndarray] = eqx.field(converter=dict)

    def __init__(
        self,
        channels: Optional[list[Channel]] = None,
        externals: Optional[list[ExternalInput]] = None,
        c=1.0,
        tadj=1.0,
        r=1.0,
        l=10.0,
        initial_state=None,
    ):
        self.channels = {c.name: c for c in channels} if channels is not None else {}
        self.externals = {e.name: e for e in externals} if externals is not None else {}
        self.initial_state = initial_state if initial_state is not None else {}

        self.c = c
        self.tadj = tadj
        self.r = r
        self.l = l

    @property
    def area(self):
        return 2 * jnp.pi * self.r * self.l  # μm²

    def __repr__(self):
        cls_name = self.__class__.__name__
        channels_str = ", ".join([c.__repr__() for c in self.channels.values()])
        externals_str = ", ".join([e.__repr__() for e in self.externals.values()])
        return f"{cls_name}(channels=[{channels_str}], externals=[{externals_str}], c={self.c})"

    def __call__(self, t, u, args=None):
        dx_dt = {}
        dx_dt_channels = {}
        aggregated_currents = {}

        i_ext = (
            self.externals["i_ext"](t, u, args) if "i_ext" in self.externals else 0.0
        )
        i = i_ext / self.area * 1e2  # pA/μm² -> mA/cm²

        for name, e in self.externals.items():
            if name in u:
                u = e(t, u, args)

        # Algebraic pre-pass: update u with any algebraically-computed states
        for name, channel in self.channels.items():
            if hasattr(channel, "precompute"):
                update = channel.precompute(t, u, args)
                if update:
                    u = {**u, **update}

        # First pass: evaluate channels that do not require aggregated current inputs.
        for name, channel in self.channels.items():
            if channel.requires:
                continue

            channel_terms = channel(t, u, args)
            dx_dt_channels[name] = channel_terms

            ion = getattr(channel, "ion", None)
            if ion is not None:
                aggregated_currents[ion] = aggregated_currents.get(
                    ion, 0.0
                ) + channel_terms.get(channel.current_name, 0.0)

        # Second pass: evaluate channels that need aggregated current inputs.
        for name, channel in self.channels.items():
            if not channel.requires:
                continue
            required_currents = tuple(
                aggregated_currents.get(req, 0.0) for req in channel.requires
            )
            dx_dt_channels[name] = channel(t, u, required_currents)

        for name, channel_terms in dx_dt_channels.items():
            current_key = self.channels[name].current_name
            i += -channel_terms.get(current_key, 0.0)
            dx_dt.update({k: v for k, v in channel_terms.items() if k != current_key})

        # dv/dt = sum(i)/c
        dx_dt["v"] = i / self.c  # * 1000.0  # mA/cm² -> μA/cm²
        return (
            dx_dt  # if unravel_fn is None else jax.flatten_util.ravel_pytree(dx_dt)[0]
        )

    def set(self, set_dict):
        return tree_set_with_path(self, set_dict)

    def init(self, t, u, args=None):
        u["v"] = u.get("v", -70.0)
        u0 = {}
        for _, channel in self.channels.items():
            channel_u0 = channel.init(t, u, args)
            u0.update({k: channel_u0[k] for k in channel_u0 if k != "v"})
        u0.update(u)  # overwrite with user provided values
        return u0

    def insert(self, component: Union[Channel, ExternalInput]):
        if isinstance(component, Channel):
            channels = self.channels.copy()
            channels.update({component.name: component})
            return eqx.tree_at(lambda x: x.channels, self, channels)
        elif isinstance(component, ExternalInput):
            externals = self.externals.copy()
            externals.update({component.name: component})
            return eqx.tree_at(lambda x: x.externals, self, externals)
        else:
            raise ValueError(f"Invalid type: {type(component)}")

    def delete(self, component_name: str):
        updated_self = self
        if component_name in self.channels:
            channels = self.channels.copy()
            channels.pop(component_name)
            updated_self = eqx.tree_at(lambda x: x.channels, self, channels)
        elif component_name in self.externals:
            externals = self.externals.copy()
            externals.pop(component_name)
            updated_self = eqx.tree_at(lambda x: x.externals, self, externals)
        else:
            raise ValueError(f"Component {component_name} not found")
        return updated_self


@functools.lru_cache(maxsize=64)
def _make_save_fn(save_dims: tuple[str, ...]):
    def save_fn(t, y, args=None):
        return {k: y[k] for k in save_dims}

    return save_fn


def integrate(
    func,
    ts,
    y0,
    save_dims=None,
    throw=True,
    rtol=1e-4,
    atol=1e-6,
    solver=None,
    adjoint=None,
    **kwargs,
):
    """Integrate an ODE and return (ts, ys).

    Default solver is Kvaerno5 (implicit, stiff-aware) — suitable for forward
    simulation.  For gradient computation with stiff L5PC-type models use an
    explicit solver (e.g. diffrax.Tsit5()) via the solver= argument, because
    differentiating through Kvaerno5's implicit linear solve compounds cotangent
    amplification by 1/min_sv per step, causing NaN gradients.
    """
    saveat = diffrax.SaveAt(ts=ts)
    if save_dims is not None:
        saveat = diffrax.SaveAt(ts=ts, fn=_make_save_fn(tuple(save_dims)))

    if solver is None:
        # SVD linear solver for robustness when (I - h*a*J) is ill-conditioned
        # (e.g. large off-diagonal Jacobian from high-conductance channels like Kv3.1).
        _default_rf = diffrax.Kvaerno5().root_finder
        solver = diffrax.Kvaerno5(
            root_finder=diffrax.VeryChord(
                rtol=_default_rf.rtol,
                atol=_default_rf.atol,
                norm=_default_rf.norm,
                linear_solver=lineax.AutoLinearSolver(well_posed=False),
            )
        )

    if adjoint is None:
        adjoint = diffrax.RecursiveCheckpointAdjoint()

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(func),
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=saveat,
        throw=throw,
        adjoint=adjoint,
        **kwargs,
    )
    return solution.ts, solution.ys


def scaled_integrate(func, ts, y0, save_dims=None, solver=None, **integrate_kwargs):
    """Wrap `integrate` with solver-level time normalization.

    User API:
      - `ts` in physical units (e.g. ms)
      - `y0` in physical units (e.g. mV)
    Returns:
      - physical `ts` and corresponding `ys`
    """
    ts = jnp.asarray(ts)
    t0 = ts[0]
    t1 = ts[-1]
    T = t1 - t0
    # avoid divide-by-zero if ts has no span
    T = jnp.where(T == 0, 1.0, T)

    taus = (ts - t0) / T  # normalized time in [0, 1]

    def func_tau(tau, y, args=None):
        t = t0 + T * tau  # physical time
        dy_dt = func(t, y, args)  # physical derivative
        return jax.tree.map(lambda x: T * x, dy_dt)  # dy/dtau

    # Call your existing integrate on τ-grid.
    # It will use dt0 = taus[1] - taus[0], PID tolerances etc.
    _, ys = integrate(
        func_tau,
        taus,
        y0,
        save_dims=save_dims,
        solver=solver,
        **integrate_kwargs,
    )

    # Return physical times with the τ-solved states
    return ts, ys
