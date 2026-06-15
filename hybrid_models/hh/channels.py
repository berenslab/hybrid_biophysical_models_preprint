import jax.numpy as jnp
import jax
import jax.random as jr
import equinox.nn as nn

import equinox as eqx
from typing import Optional, Union
import warnings

from ..utils import safe_exp, safe_expm1, block_stack, tree_set_with_path, safe_log, vtrap, efun


class Channel(eqx.Module):
    name: str = eqx.field(static=True)
    ion: Optional[str] = eqx.field(static=True)
    requires: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, name: Optional[str] = None):
        self.name = self.__class__.__name__.lower() if name is None else name
        self.ion = getattr(type(self), "ion", None)
        self.requires = getattr(type(self), "requires", ())

    @property
    def current_name(self):
        return f"i_{self.name}"

    def i(self, t, u, args=None):
        return 0.0

    def __call__(self, t, u, args=None):
        # should also return current. Avoids having to call the
        # Neural ODE twice or cache the output somehow.
        return {self.current_name: self.i(t, u, args)}

    def xinf(self, t, u, args=None):
        return {}

    def tau(self, t, u, args=None):
        return {}

    def init(self, t, u, args=None):
        return {}

    def set(self, set_dict):
        return tree_set_with_path(self, set_dict)

    def __repr__(self):
        cls_name = self.__class__.__name__
        fields = self.__dataclass_fields__.keys()
        params = {k: getattr(self, k) for k in fields}
        del params["name"]
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        return f"{cls_name}({params_str})"


class Leak(Channel):
    gl: jnp.ndarray = eqx.field(converter=jnp.array)
    el: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gl=0.1, el=-70.0, tadj=1.0):
        super().__init__()
        self.gl = gl
        self.el = el
        self.tadj = tadj

    def i(self, t, u, args=None):
        v = u["v"]
        return self.gl * (v - self.el)

    def __call__(self, t, u, args=None):
        return {f"i_{self.name}": self.i(t, u, args)}

    def init(self, t, u, args=None):
        return {}


class Na(Channel):
    gna: jnp.ndarray = eqx.field(converter=jnp.array)
    ena: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gna=120.0, ena=50.0, tadj=1.0):
        super().__init__()
        self.gna = gna
        self.ena = ena
        self.tadj = tadj

    def alpha(self, v):
        alpha_m = 0.1 * vtrap(-(v + 40.0), 10.0)
        alpha_h = 0.07 * safe_exp(-(v + 65.0) / 20.0)
        return {"m": alpha_m, "h": alpha_h}

    def beta(self, v):
        beta_m = 4.0 * safe_exp(-(v + 65.0) / 18.0)
        beta_h = 1.0 / (1.0 + safe_exp(-(v + 35.0) / 10.0))
        return {"m": beta_m, "h": beta_h}

    def tau(self, t, u, args=None):
        alpha = self.alpha(u["v"])
        beta = self.beta(u["v"])
        tau_m = 1.0 / (alpha["m"] + beta["m"])
        tau_h = 1.0 / (alpha["h"] + beta["h"])
        return {"m": tau_m, "h": tau_h}

    def xinf(self, t, u, args=None):
        alpha = self.alpha(u["v"])
        beta = self.beta(u["v"])
        m_inf = alpha["m"] / (alpha["m"] + beta["m"])
        h_inf = alpha["h"] / (alpha["h"] + beta["h"])
        return {"m": m_inf, "h": h_inf}

    def i(self, t, u, args=None):
        v, m, h = u["v"], u["m"], u["h"]
        return self.gna * m**3 * h * (v - self.ena)

    def __call__(self, t, u, args=None):
        v, m, h = u["v"], u["m"], u["h"]

        alpha = self.alpha(v)
        beta = self.beta(v)

        dm_dt = self.tadj * (alpha["m"] * (1.0 - m) - beta["m"] * m)
        dh_dt = self.tadj * (alpha["h"] * (1.0 - h) - beta["h"] * h)

        return {f"i_{self.name}": self.i(t, u, args), "m": dm_dt, "h": dh_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            v = u["v"]
            alpha = self.alpha(v)
            beta = self.beta(v)
            m0 = alpha["m"] / (alpha["m"] + beta["m"])
            h0 = alpha["h"] / (alpha["h"] + beta["h"])
            return {"m": m0, "h": h0}
        return {"m": 0.1, "h": 0.1}


class K(Channel):
    gk: jnp.ndarray = eqx.field(converter=jnp.array)
    ek: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gk=36.0, ek=-75.0, tadj=1.0):
        super().__init__()
        self.gk = gk
        self.ek = ek
        self.tadj = tadj

    def alpha(self, v):
        return {"n": 0.01 * vtrap(-(v + 55.0), 10.0)}

    def beta(self, v):
        return {"n": 0.125 * safe_exp(-(v + 65.0) / 80.0)}

    def xinf(self, t, u, args=None):
        alpha = self.alpha(u["v"])
        beta = self.beta(u["v"])
        n_inf = alpha["n"] / (alpha["n"] + beta["n"])
        return {"n": n_inf}

    def tau(self, t, u, args=None):
        alpha = self.alpha(u["v"])
        beta = self.beta(u["v"])
        tau_n = 1.0 / (alpha["n"] + beta["n"])
        return {"n": tau_n}

    def i(self, t, u, args=None):
        v, n = u["v"], u["n"]
        return self.gk * n**4 * (v - self.ek)

    def __call__(self, t, u, args=None):
        v, n = u["v"], u["n"]

        alpha = self.alpha(v)
        beta = self.beta(v)
        dn_dt = self.tadj * (alpha["n"] * (1.0 - n) - beta["n"] * n)
        return {f"i_{self.name}": self.i(t, u, args), "n": dn_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            v = u["v"]
            alpha = self.alpha(v)
            beta = self.beta(v)
            n0 = alpha["n"] / (alpha["n"] + beta["n"])
            return {"n": n0}
        return {"n": 0.1}


class KM(Channel):
    gkm: jnp.ndarray = eqx.field(converter=jnp.array)
    ekm: jnp.ndarray = eqx.field(converter=jnp.array)
    tau_m: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gkm=1.0, ekm=-90.0, tau_m=1000.0, tadj=1.0):
        super().__init__()
        self.gkm = gkm
        self.ekm = ekm
        self.tau_m = tau_m
        self.tadj = tadj

    def xinf(self, v):
        return {"p": 1.0 / (1.0 + safe_exp(-(v + 35.0) / 10.0))}

    def tau(self, v):
        return {
            "p": self.tau_m
            / (3.3 * safe_exp((v + 35.0) / 20.0) + safe_exp(-(v + 35.0) / 20.0))
        }

    def i(self, t, u, args=None):
        v, p = u["v"], u["p"]
        return self.gkm * p * (v - self.ekm)

    def __call__(self, t, u, args=None):
        v, p = u["v"], u["p"]
        xinf = self.xinf(t, u, args=args)
        tau = self.tau(t, u, args=args)
        dp_dt = (xinf["p"] - p) / (tau["p"] / self.tadj)
        return {f"i_{self.name}": self.i(t, u, args), "p": dp_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"p": xinf["p"]}
        return {"p": 0.1}


class Pospischil_Leak(Channel):
    """Leak current based on Pospischil et al., 2008."""

    gl: jnp.ndarray = eqx.field(converter=jnp.array)
    el: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gl=0.1, el=-70.0, tadj=1.0):
        super().__init__()
        self.gl = gl
        self.el = el
        self.tadj = tadj

    def i(self, t, u, args=None):
        v = u["v"]
        return self.gl * (v - self.el)

    def __call__(self, t, u, args=None):
        return {f"i_{self.name}": self.i(t, u, args)}

    def init(self, t, u, args=None):
        return {}


class Pospischil_Na(Channel):
    """Sodium channel based on Pospischil et al., 2008."""

    gna: jnp.ndarray = eqx.field(converter=jnp.array)
    ena: jnp.ndarray = eqx.field(converter=jnp.array)
    vt: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gna=50.0, ena=50.0, vt=-60.0, tadj=1.0):
        super().__init__()
        self.gna = gna
        self.ena = ena
        self.vt = vt
        self.tadj = tadj

    def m_gate(self, v):
        v_alpha = v - self.vt - 13.0
        alpha = 0.32 * efun(-0.25 * v_alpha) / 0.25

        v_beta = v - self.vt - 40.0
        beta = 0.28 * efun(0.2 * v_beta) / 0.2
        return alpha, beta

    def h_gate(self, v):
        v_alpha = v - self.vt - 17.0
        alpha = 0.128 * safe_exp(-v_alpha / 18.0)

        v_beta = v - self.vt - 40.0
        beta = 4.0 / (safe_exp(-v_beta / 5.0) + 1.0)
        return alpha, beta

    def xinf(self, t, u, args=None):
        alpha_m, beta_m = self.m_gate(u["v"])
        alpha_h, beta_h = self.h_gate(u["v"])
        m_inf = alpha_m / (alpha_m + beta_m)
        h_inf = alpha_h / (alpha_h + beta_h)
        return {"m": m_inf, "h": h_inf}

    def tau(self, t, u, args=None):
        alpha_m, beta_m = self.m_gate(u["v"])
        alpha_h, beta_h = self.h_gate(u["v"])
        tau_m = 1.0 / (alpha_m + beta_m)
        tau_h = 1.0 / (alpha_h + beta_h)
        return {"m": tau_m / self.tadj, "h": tau_h / self.tadj}

    def i(self, t, u, args=None):
        v, m, h = u["v"], u["m"], u["h"]
        return self.gna * m**3 * h * (v - self.ena)

    def __call__(self, t, u, args=None):
        v, m, h = u["v"], u["m"], u["h"]
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)

        dm_dt = self.tadj * (alpha_m * (1.0 - m) - beta_m * m)
        dh_dt = self.tadj * (alpha_h * (1.0 - h) - beta_h * h)

        return {f"i_{self.name}": self.i(t, u, args), "m": dm_dt, "h": dh_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            v = u["v"]
            alpha_m, beta_m = self.m_gate(v)
            alpha_h, beta_h = self.h_gate(v)
            m0 = alpha_m / (alpha_m + beta_m)
            h0 = alpha_h / (alpha_h + beta_h)
            return {"m": m0, "h": h0}
        return {"m": 0.1, "h": 0.1}


class Pospischil_K(Channel):
    """Potassium channel based on Pospischil et al., 2008."""

    gk: jnp.ndarray = eqx.field(converter=jnp.array)
    ek: jnp.ndarray = eqx.field(converter=jnp.array)
    vt: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gk=5.0, ek=-90.0, vt=-60.0, tadj=1.0):
        super().__init__()
        self.gk = gk
        self.ek = ek
        self.vt = vt
        self.tadj = tadj

    def n_gate(self, v):
        v_alpha = v - self.vt - 15.0
        alpha = 0.032 * efun(-0.2 * v_alpha) / 0.2

        v_beta = v - self.vt - 10.0
        beta = 0.5 * safe_exp(-v_beta / 40.0)
        return alpha, beta

    def xinf(self, t, u, args=None):
        alpha_n, beta_n = self.n_gate(u["v"])
        n_inf = alpha_n / (alpha_n + beta_n)
        return {"n": n_inf}

    def tau(self, t, u, args=None):
        alpha_n, beta_n = self.n_gate(u["v"])
        tau_n = 1.0 / (alpha_n + beta_n) / self.tadj
        return {"n": tau_n}

    def i(self, t, u, args=None):
        v, n = u["v"], u["n"]
        return self.gk * n**4 * (v - self.ek)

    def __call__(self, t, u, args=None):
        v, n = u["v"], u["n"]
        alpha_n, beta_n = self.n_gate(v)

        dn_dt = self.tadj * (alpha_n * (1.0 - n) - beta_n * n)
        return {f"i_{self.name}": self.i(t, u, args), "n": dn_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            v = u["v"]
            alpha_n, beta_n = self.n_gate(v)
            n0 = alpha_n / (alpha_n + beta_n)
            return {"n": n0}
        return {"n": 0.1}


class Pospischil_Km(Channel):
    """Slow M Potassium channel based on Pospischil et al., 2008."""

    gkm: jnp.ndarray = eqx.field(converter=jnp.array)
    ekm: jnp.ndarray = eqx.field(converter=jnp.array)
    taumax: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gkm=0.004, ekm=-90.0, taumax=4000.0, tadj=1.0):
        super().__init__()
        self.gkm = gkm
        self.ekm = ekm
        self.taumax = taumax
        self.tadj = tadj

    def xinf(self, t, u, args=None):
        v_p = u["v"] + 35.0
        p_inf = 1.0 / (1.0 + safe_exp(-0.1 * v_p))
        return {"p": p_inf}

    def tau(self, t, u, args=None):
        v_p = u["v"] + 35.0
        tau_p = self.taumax / (3.3 * safe_exp(0.05 * v_p) + safe_exp(-0.05 * v_p))
        return {"p": tau_p / self.tadj}

    def i(self, t, u, args=None):
        v, p = u["v"], u["p"]
        return self.gkm * p * (v - self.ekm)

    def __call__(self, t, u, args=None):
        v, p = u["v"], u["p"]
        xinf = self.xinf(t, u, args=args)
        tau = self.tau(t, u, args=args)

        dp_dt = (xinf["p"] - p) / (tau["p"] / self.tadj)
        return {f"i_{self.name}": self.i(t, u, args), "p": dp_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"p": xinf["p"]}
        return {"p": 0.1}


class Pospischil_CaL(Channel):
    """L-type Calcium channel based on Pospischil et al., 2008."""

    ion = "ca"

    gcal: jnp.ndarray = eqx.field(converter=jnp.array)
    eca: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gcal=0.1, eca=120.0, tadj=1.0):
        super().__init__()
        self.gcal = gcal
        self.eca = eca
        self.tadj = tadj

    def q_gate(self, v):
        v_alpha = -v - 27.0
        alpha = 0.055 * efun(v_alpha / 3.8) * 3.8

        v_beta = -v - 75.0
        beta = 0.94 * safe_exp(v_beta / 17.0)
        return alpha, beta

    def r_gate(self, v):
        v_alpha = -v - 13.0
        alpha = 0.000457 * safe_exp(v_alpha / 50.0)

        v_beta = -v - 15.0
        beta = 0.0065 / (safe_exp(v_beta / 28.0) + 1.0)
        return alpha, beta

    def xinf(self, t, u, args=None):
        alpha_q, beta_q = self.q_gate(u["v"])
        alpha_r, beta_r = self.r_gate(u["v"])
        q_inf = alpha_q / (alpha_q + beta_q)
        r_inf = alpha_r / (alpha_r + beta_r)
        return {"q": q_inf, "r": r_inf}

    def tau(self, t, u, args=None):
        alpha_q, beta_q = self.q_gate(u["v"])
        alpha_r, beta_r = self.r_gate(u["v"])
        tau_q = 1.0 / (alpha_q + beta_q)
        tau_r = 1.0 / (alpha_r + beta_r)
        return {"q": tau_q / self.tadj, "r": tau_r / self.tadj}

    def i(self, t, u, args=None):
        v, q, r = u["v"], u["q"], u["r"]
        return self.gcal * q**2 * r * (v - self.eca)

    def __call__(self, t, u, args=None):
        v, q, r = u["v"], u["q"], u["r"]
        alpha_q, beta_q = self.q_gate(v)
        alpha_r, beta_r = self.r_gate(v)

        dq_dt = self.tadj * (alpha_q * (1.0 - q) - beta_q * q)
        dr_dt = self.tadj * (alpha_r * (1.0 - r) - beta_r * r)

        return {f"i_{self.name}": self.i(t, u, args), "q": dq_dt, "r": dr_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"q": xinf["q"], "r": xinf["r"]}
        return {"q": 0.1, "r": 0.1}


class Pospischil_CaT(Channel):
    """T-type Calcium channel based on Pospischil et al., 2008."""

    ion = "ca"

    gcat: jnp.ndarray = eqx.field(converter=jnp.array)
    eca: jnp.ndarray = eqx.field(converter=jnp.array)
    vx: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gcat=0.04, eca=120.0, vx=2.0, tadj=1.0):
        super().__init__()
        self.gcat = gcat
        self.eca = eca
        self.vx = vx
        self.tadj = tadj

    def xinf(self, t, u, args=None):
        v_u1 = u["v"] + self.vx + 81.0
        u_inf = 1.0 / (1.0 + safe_exp(v_u1 / 4.0))
        return {"u": u_inf}

    def tau(self, t, u, args=None):
        v_u1 = u["v"] + self.vx + 81.0
        tau_u = (30.8 + (211.4 + safe_exp((u["v"] + self.vx + 113.2) / 5.0))) / (
            3.7 * (1.0 + safe_exp((u["v"] + self.vx + 84.0) / 3.2))
        )
        return {"u": tau_u / self.tadj}

    def i(self, t, u, args=None):
        v, u_gate = u["v"], u["u"]
        s_inf = 1.0 / (1.0 + safe_exp(-(v + self.vx + 57.0) / 6.2))
        return self.gcat * (s_inf**2) * u_gate * (v - self.eca)

    def __call__(self, t, u, args=None):
        v, u_gate = u["v"], u["u"]
        xinf = self.xinf(t, u, args=args)
        tau = self.tau(t, u, args=args)

        du_dt = (xinf["u"] - u_gate) / (tau["u"] / self.tadj)
        return {f"i_{self.name}": self.i(t, u, args), "u": du_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"u": xinf["u"]}
        return {"u": 0.1}


class Allen_Nap(Channel):
    """Persistent Na+ channel (Nap) from Magistretti & Alonso 1999, Allen Institute.
    Instantaneous activation m_inf, one inactivation gate h_nap. Q10 via tadj.
    h steady-state uses mod closed-form hInf = 1/(1+exp((v+48.8)/10)); tau from rates."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ena: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.01, ena=50.0, tadj=1.0):
        super().__init__()
        self.gbar = gbar
        self.ena = ena
        self.tadj = tadj

    def m_inf(self, v):
        return 1.0 / (1.0 + safe_exp(-(v + 52.6) / 4.6))

    def h_inf(self, v):
        """Closed-form h steady-state from mod: 1/(1+exp((v - -48.8)/10))."""
        return 1.0 / (1.0 + safe_exp((v + 48.8) / 10.0))

    def h_gate(self, v):
        alpha_h = 2.88e-6 * vtrap(v + 17.0, 4.63)
        beta_h = 6.94e-6 * vtrap(-(v + 64.4), 2.63)
        return alpha_h, beta_h

    def xinf(self, t, u, args=None):
        v = u["v"]
        return {"h_nap": self.h_inf(v)}

    def tau(self, t, u, args=None):
        v = u["v"]
        alpha_h, beta_h = self.h_gate(v)
        return {"h_nap": 1.0 / (alpha_h + beta_h) / self.tadj}

    def i(self, t, u, args=None):
        v, h_nap = u["v"], u["h_nap"]
        return self.gbar * self.m_inf(v) * h_nap * (v - self.ena)

    def __call__(self, t, u, args=None):
        v, h_nap = u["v"], u["h_nap"]
        alpha_h, beta_h = self.h_gate(v)
        tau_h = 1.0 / (alpha_h + beta_h) / self.tadj
        dh_dt = (self.h_inf(v) - h_nap) / tau_h
        return {f"i_{self.name}": self.i(t, u, args), "h_nap": dh_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"h_nap": xinf["h_nap"]}
        return {"h_nap": 0.1}


class Allen_Ca_HVA(Channel):
    """HVA Ca2+ channel, Reuveni et al. 1993. States: ca_hva_m, ca_hva_h. g = gbar*m*m*h. No Q10 in mod; tadj=1."""

    ion = "ca"

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    eca: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.01, eca=120.0, tadj=1.0):
        super().__init__()
        self.gbar = gbar
        self.eca = eca
        self.tadj = tadj

    def m_gate(self, v):
        alpha = 0.055 * vtrap(-27.0 - v, 3.8)
        beta = 0.94 * safe_exp((-75.0 - v) / 17.0)
        return alpha, beta

    def h_gate(self, v):
        alpha = 0.000457 * safe_exp((-13.0 - v) / 50.0)
        beta = 0.0065 / (safe_exp((-v - 15.0) / 28.0) + 1.0)
        return alpha, beta

    def xinf(self, t, u, args=None):
        v = u["v"]
        am, bm = self.m_gate(v)
        ah, bh = self.h_gate(v)
        return {"ca_hva_m": am / (am + bm), "ca_hva_h": ah / (ah + bh)}

    def tau(self, t, u, args=None):
        v = u["v"]
        am, bm = self.m_gate(v)
        ah, bh = self.h_gate(v)
        return {
            "ca_hva_m": 1.0 / (am + bm),
            "ca_hva_h": 1.0 / (ah + bh),
        }

    def i(self, t, u, args=None):
        v, m, h = u["v"], u["ca_hva_m"], u["ca_hva_h"]
        eca = u.get("eca", self.eca)
        return self.gbar * m * m * h * (v - eca)

    def __call__(self, t, u, args=None):
        v, m, h = u["v"], u["ca_hva_m"], u["ca_hva_h"]
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)
        dm_dt = alpha_m * (1.0 - m) - beta_m * m
        dh_dt = alpha_h * (1.0 - h) - beta_h * h
        return {
            f"i_{self.name}": self.i(t, u, args),
            "ca_hva_m": dm_dt,
            "ca_hva_h": dh_dt,
        }

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"ca_hva_m": xinf["ca_hva_m"], "ca_hva_h": xinf["ca_hva_h"]}
        return {"ca_hva_m": 0.1, "ca_hva_h": 0.1}


class Allen_Ca_LVA(Channel):
    """LVA Ca2+ channel, Avery & Johnston 1996, Randall 1997. Q10 via tadj. v shifted -10 mV. States: ca_lva_m, ca_lva_h."""

    ion = "ca"

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    eca: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.01, eca=120.0, tadj=1.0):
        super().__init__()
        self.gbar = gbar
        self.eca = eca
        self.tadj = tadj

    def xinf(self, t, u, args=None):
        v = u["v"]
        v_ = v + 10.0
        m_inf = 1.0 / (1.0 + safe_exp((v_ - (-30.0)) / (-6.0)))
        h_inf = 1.0 / (1.0 + safe_exp((v_ - (-80.0)) / 6.4))
        return {"ca_lva_m": m_inf, "ca_lva_h": h_inf}

    def tau(self, t, u, args=None):
        v = u["v"]
        v_ = v + 10.0
        m_tau_raw = 5.0 + 20.0 / (1.0 + safe_exp((v_ - (-25.0)) / 5.0))
        h_tau_raw = 20.0 + 50.0 / (1.0 + safe_exp((v_ - (-40.0)) / 7.0))
        return {"ca_lva_m": m_tau_raw / self.tadj, "ca_lva_h": h_tau_raw / self.tadj}

    def i(self, t, u, args=None):
        v, m, h = u["v"], u["ca_lva_m"], u["ca_lva_h"]
        eca = u.get("eca", self.eca)
        return self.gbar * m * m * h * (v - eca)

    def __call__(self, t, u, args=None):
        v, m, h = u["v"], u["ca_lva_m"], u["ca_lva_h"]
        xinf = self.xinf(t, u, args=args)
        tau = self.tau(t, u, args=args)
        dm_dt = (xinf["ca_lva_m"] - m) / tau["ca_lva_m"]
        dh_dt = (xinf["ca_lva_h"] - h) / tau["ca_lva_h"]
        return {
            f"i_{self.name}": self.i(t, u, args),
            "ca_lva_m": dm_dt,
            "ca_lva_h": dh_dt,
        }

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"ca_lva_m": xinf["ca_lva_m"], "ca_lva_h": xinf["ca_lva_h"]}
        return {"ca_lva_m": 0.1, "ca_lva_h": 0.1}


class Allen_Ih(Channel):
    """HCN / Ih channel, Kole et al. 2006. Non-specific, reversal ehcn. State: ih_m. No Q10 in mod; tadj=1."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ehcn: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.01, ehcn=-45.0, tadj=1.0):
        super().__init__()
        self.gbar = gbar
        self.ehcn = ehcn
        self.tadj = tadj

    def m_gate(self, v):
        # alpha = 0.001 * 6.43 * vtrap(v + 154.9, 11.9) # allen
        alpha = 0.001 * 6.43 * vtrap(v + 154.9 + 1e-6, 11.9)  # jaxley
        beta = 0.001 * 193.0 * safe_exp(v / 33.1)
        return alpha, beta

    def xinf(self, t, u, args=None):
        v = u["v"]
        a, b = self.m_gate(v)
        return {"ih_m": a / (a + b)}

    def tau(self, t, u, args=None):
        v = u["v"]
        a, b = self.m_gate(v)
        return {"ih_m": 1.0 / (a + b)}

    def i(self, t, u, args=None):
        v, m = u["v"], u["ih_m"]
        return self.gbar * m * (v - self.ehcn)

    def __call__(self, t, u, args=None):
        v, m = u["v"], u["ih_m"]
        alpha, beta = self.m_gate(v)
        dm_dt = alpha * (1.0 - m) - beta * m
        return {f"i_{self.name}": self.i(t, u, args), "ih_m": dm_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"ih_m": xinf["ih_m"]}
        return {"ih_m": 0.1}


class Allen_Im(Channel):
    """M-current, Adams et al. 1982. State: im_m. Q10 via tadj."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ek: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.01, ek=-77.0, tadj=1.0):
        super().__init__()
        self.gbar = gbar
        self.ek = ek
        self.tadj = tadj

    def m_gate(self, v):
        alpha = 3.3e-3 * safe_exp(2.5 * 0.04 * (v - (-35.0)))
        beta = 3.3e-3 * safe_exp(-2.5 * 0.04 * (v - (-35.0)))
        return alpha, beta

    def xinf(self, t, u, args=None):
        v = u["v"]
        a, b = self.m_gate(v)
        return {"im_m": a / (a + b)}

    def tau(self, t, u, args=None):
        v = u["v"]
        a, b = self.m_gate(v)
        return {"im_m": 1.0 / (a + b) / self.tadj}

    def i(self, t, u, args=None):
        v, m = u["v"], u["im_m"]
        return self.gbar * m * (v - self.ek)

    def __call__(self, t, u, args=None):
        v, m = u["v"], u["im_m"]
        alpha, beta = self.m_gate(v)
        dm_dt = self.tadj * (alpha * (1.0 - m) - beta * m)
        return {f"i_{self.name}": self.i(t, u, args), "im_m": dm_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"im_m": xinf["im_m"]}
        return {"im_m": 0.1}


class Allen_Im_v2(Channel):
    """M-current v2, Vervaeke et al. 2006. State: im_v2_m. Q10 from 30 C. Tau = (15 + 1/(a+b))/tadj."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ek: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.01, ek=-77.0, tadj=1.0):
        super().__init__()
        self.gbar = gbar
        self.ek = ek
        self.tadj = tadj

    def m_gate(self, v):
        alpha = 0.007 * safe_exp(6.0 * 0.4 * (v - (-48.0)) / 26.12)
        beta = 0.007 * safe_exp(-6.0 * (1.0 - 0.4) * (v - (-48.0)) / 26.12)
        return alpha, beta

    def xinf(self, t, u, args=None):
        v = u["v"]
        a, b = self.m_gate(v)
        return {"im_v2_m": a / (a + b)}

    def tau(self, t, u, args=None):
        v = u["v"]
        a, b = self.m_gate(v)
        tau_raw = 15.0 + 1.0 / (a + b)
        return {"im_v2_m": tau_raw / self.tadj}

    def i(self, t, u, args=None):
        v, m = u["v"], u["im_v2_m"]
        return self.gbar * m * (v - self.ek)

    def __call__(self, t, u, args=None):
        v, m = u["v"], u["im_v2_m"]
        xinf = self.xinf(t, u, args=args)
        tau = self.tau(t, u, args=args)
        dm_dt = (xinf["im_v2_m"] - m) / tau["im_v2_m"]
        return {f"i_{self.name}": self.i(t, u, args), "im_v2_m": dm_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"im_v2_m": xinf["im_v2_m"]}
        return {"im_v2_m": 0.1}


class Allen_K_P(Channel):
    """Persistent K+ (K_P), Korngreen & Sakmann 2000. States: k_p_m, k_p_h. Q10 via tadj."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ek: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)
    vshift: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.01, ek=-77.0, tadj=1.0, vshift=0.0):
        super().__init__()
        self.gbar = gbar
        self.ek = ek
        self.tadj = tadj
        self.vshift = vshift

    def xinf(self, t, u, args=None):
        v = u["v"]
        v_adj = v - self.vshift
        # m_inf = 1.0 / (1.0 + safe_exp(-(v_adj + 14.3) / 14.6)) # allen
        m_inf = 1.0 / (1.0 + safe_exp(-(v_adj + 1.0) / 12.0))  # jaxley
        h_inf = 1.0 / (1.0 + safe_exp(-(v_adj + 54.0) / (-11.0)))
        return {"k_p_m": m_inf, "k_p_h": h_inf}

    def tau(self, t, u, args=None):
        v = u["v"]
        v_adj = v - self.vshift
        m_tau = jnp.where(
            v_adj < -50.0,
            1.0 * (1.25 + 175.03 * safe_exp(v_adj * 0.026)),
            1.0 * (1.25 + 13.0 * safe_exp(-v_adj * 0.026)),
        )
        h_tau = 360.0 + (1010.0 + 24.0 * (v_adj + 55.0)) * safe_exp(
            -(((v_adj + 75.0) / 48.0) ** 2)
        )
        return {"k_p_m": m_tau / self.tadj, "k_p_h": h_tau / self.tadj}

    def i(self, t, u, args=None):
        v, m, h = u["v"], u["k_p_m"], u["k_p_h"]
        return self.gbar * m * m * h * (v - self.ek)

    def __call__(self, t, u, args=None):
        v, m, h = u["v"], u["k_p_m"], u["k_p_h"]
        xinf = self.xinf(t, u, args=args)
        tau = self.tau(t, u, args=args)
        dm_dt = (xinf["k_p_m"] - m) / tau["k_p_m"]
        dh_dt = (xinf["k_p_h"] - h) / tau["k_p_h"]
        return {f"i_{self.name}": self.i(t, u, args), "k_p_m": dm_dt, "k_p_h": dh_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"k_p_m": xinf["k_p_m"], "k_p_h": xinf["k_p_h"]}
        return {"k_p_m": 0.1, "k_p_h": 0.1}


class Allen_K_T(Channel):
    """Transient K+ (K_T), Korngreen & Sakmann 2000. States: k_t_m, k_t_h. g = gbar*m^4*h. Q10 via tadj."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ek: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)
    vshift: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.01, ek=-77.0, tadj=1.0, vshift=0.0):
        super().__init__()
        self.gbar = gbar
        self.ek = ek
        self.tadj = tadj
        self.vshift = vshift

    def xinf(self, t, u, args=None):
        v = u["v"]
        v_adj = v - self.vshift
        # m_inf = 1.0 / (1.0 + safe_exp(-(v_adj + 47.0) / 29.0)) # allen
        m_inf = 1 / (1 + safe_exp(-(v_adj + 0.0) / 19.0))  # jaxley
        h_inf = 1.0 / (1.0 + safe_exp(-(v_adj + 66.0) / (-10.0)))
        return {"k_t_m": m_inf, "k_t_h": h_inf}

    def tau(self, t, u, args=None):
        v = u["v"]
        v_adj = v - self.vshift
        m_tau_raw = 0.34 + 1.0 * 0.92 * safe_exp(-(((v_adj + 71.0) / 59.0) ** 2))
        h_tau_raw = 8.0 + 1.0 * 49.0 * safe_exp(-(((v_adj + 73.0) / 23.0) ** 2))
        return {"k_t_m": m_tau_raw / self.tadj, "k_t_h": h_tau_raw / self.tadj}

    def i(self, t, u, args=None):
        v, m, h = u["v"], u["k_t_m"], u["k_t_h"]
        return self.gbar * m**4 * h * (v - self.ek)

    def __call__(self, t, u, args=None):
        v, m, h = u["v"], u["k_t_m"], u["k_t_h"]
        xinf = self.xinf(t, u, args=args)
        tau = self.tau(t, u, args=args)
        dm_dt = (xinf["k_t_m"] - m) / tau["k_t_m"]
        dh_dt = (xinf["k_t_h"] - h) / tau["k_t_h"]
        return {f"i_{self.name}": self.i(t, u, args), "k_t_m": dm_dt, "k_t_h": dh_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"k_t_m": xinf["k_t_m"], "k_t_h": xinf["k_t_h"]}
        return {"k_t_m": 0.1, "k_t_h": 0.1}


class Allen_Kd(Channel):
    """Kd channel, Foust et al. 2011. States: kd_m, kd_h. g = gbar*m*h. Constant tau m=1, h=1500 ms."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ek: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.01, ek=-77.0, tadj=1.0):
        super().__init__()
        self.gbar = gbar
        self.ek = ek
        self.tadj = tadj

    def xinf(self, t, u, args=None):
        v = u["v"]
        m_inf = 1.0 - 1.0 / (1.0 + safe_exp((v - (-43.0)) / 8.0))
        h_inf = 1.0 / (1.0 + safe_exp((v - (-67.0)) / 7.3))
        return {"kd_m": m_inf, "kd_h": h_inf}

    def tau(self, t, u, args=None):
        return {"kd_m": 1.0, "kd_h": 1500.0}

    def i(self, t, u, args=None):
        v, m, h = u["v"], u["kd_m"], u["kd_h"]
        return self.gbar * m * h * (v - self.ek)

    def __call__(self, t, u, args=None):
        v, m, h = u["v"], u["kd_m"], u["kd_h"]
        xinf = self.xinf(t, u, args=args)
        tau = self.tau(t, u, args=args)
        dm_dt = (xinf["kd_m"] - m) / tau["kd_m"]
        dh_dt = (xinf["kd_h"] - h) / tau["kd_h"]
        return {f"i_{self.name}": self.i(t, u, args), "kd_m": dm_dt, "kd_h": dh_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"kd_m": xinf["kd_m"], "kd_h": xinf["kd_h"]}
        return {"kd_m": 0.1, "kd_h": 0.1}


class Allen_Kv2like(Channel):
    """Kv2-like K+ channel, Keren et al. 2005 / Liu & Bean 2014. States: kv2_m, kv2_h1, kv2_h2. g = gbar*m*m*(0.5*h1+0.5*h2). Q10 via tadj."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ek: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.01, ek=-77.0, tadj=1.0):
        super().__init__()
        self.gbar = gbar
        self.ek = ek
        self.tadj = tadj

    def m_gate(self, v):
        alpha = 0.12 * vtrap(-(v - 43.0), 11.0)
        beta = 0.02 * safe_exp(-(v + 1.27) / 120.0)
        return alpha, beta

    def xinf(self, t, u, args=None):
        v = u["v"]
        am, bm = self.m_gate(v)
        h_inf = 1.0 / (1.0 + safe_exp((v + 58.0) / 11.0))
        return {"kv2_m": am / (am + bm), "kv2_h1": h_inf, "kv2_h2": h_inf}

    def tau(self, t, u, args=None):
        v = u["v"]
        am, bm = self.m_gate(v)
        m_tau_raw = 2.5 / (am + bm)
        h1_tau_raw = 360.0 + (1010.0 + 23.7 * (v + 54.0)) * safe_exp(
            -(((v + 75.0) / 48.0) ** 2)
        )
        h2_tau_raw = jnp.maximum(
            2350.0 + 1380.0 * safe_exp(-0.011 * v) - 210.0 * safe_exp(-0.03 * v),
            1e-3,
        )
        return {
            "kv2_m": m_tau_raw / self.tadj,
            "kv2_h1": h1_tau_raw / self.tadj,
            "kv2_h2": h2_tau_raw / self.tadj,
        }

    def i(self, t, u, args=None):
        v, m, h1, h2 = u["v"], u["kv2_m"], u["kv2_h1"], u["kv2_h2"]
        return self.gbar * m * m * (0.5 * h1 + 0.5 * h2) * (v - self.ek)

    def __call__(self, t, u, args=None):
        v, m, h1, h2 = u["v"], u["kv2_m"], u["kv2_h1"], u["kv2_h2"]
        alpha, beta = self.m_gate(v)
        xinf = self.xinf(t, u, args=args)
        tau = self.tau(t, u, args=args)
        dm_dt = self.tadj * (alpha * (1.0 - m) - beta * m)
        dh1_dt = (xinf["kv2_h1"] - h1) / tau["kv2_h1"]
        dh2_dt = (xinf["kv2_h2"] - h2) / tau["kv2_h2"]
        return {
            f"i_{self.name}": self.i(t, u, args),
            "kv2_m": dm_dt,
            "kv2_h1": dh1_dt,
            "kv2_h2": dh2_dt,
        }

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {
                "kv2_m": xinf["kv2_m"],
                "kv2_h1": xinf["kv2_h1"],
                "kv2_h2": xinf["kv2_h2"],
            }
        return {"kv2_m": 0.1, "kv2_h1": 0.1, "kv2_h2": 0.1}


class Allen_Kv3_1(Channel):
    """Kv3.1-like K+ channel. State: kv3_m. No Q10 in mod."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ek: jnp.ndarray = eqx.field(converter=jnp.array)
    vshift: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.01, ek=-77.0, vshift=0.0, tadj=1.0):
        super().__init__()
        self.gbar = gbar
        self.ek = ek
        self.vshift = vshift
        self.tadj = tadj

    def xinf(self, t, u, args=None):
        v = u["v"]
        vs = self.vshift
        m_inf = 1.0 / (1.0 + safe_exp((v - (18.7 + vs)) / (-9.7)))
        return {"kv3_m": m_inf}

    def tau(self, t, u, args=None):
        v = u["v"]
        vs = self.vshift
        m_tau = 0.2 * 20.0 / (1.0 + safe_exp((v - (-46.56 + vs)) / (-44.14)))
        return {"kv3_m": m_tau}

    def i(self, t, u, args=None):
        v, m = u["v"], u["kv3_m"]
        return self.gbar * m * (v - self.ek)

    def __call__(self, t, u, args=None):
        v, m = u["v"], u["kv3_m"]
        xinf = self.xinf(t, u, args=args)
        tau = self.tau(t, u, args=args)
        dm_dt = (xinf["kv3_m"] - m) / tau["kv3_m"]
        return {f"i_{self.name}": self.i(t, u, args), "kv3_m": dm_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"kv3_m": xinf["kv3_m"]}
        return {"kv3_m": 0.1}


class Allen_NaTa(Channel):
    """Na+ channel (axon), Colbert & Pan 2002. States: nata_m, nata_h. g = gbar*m^3*h. Q10 via tadj."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ena: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)
    mvhalf: jnp.ndarray = eqx.field(converter=jnp.array)
    hvhalf: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(
        self,
        gbar=0.01,
        ena=50.0,
        tadj=1.0,
        mvhalf=-48.0,
        hvhalf=-69.0,
    ):
        super().__init__()
        self.gbar = gbar
        self.ena = ena
        self.tadj = tadj
        self.mvhalf = mvhalf
        self.hvhalf = hvhalf

    def m_gate(self, v):
        alpha = 0.182 * vtrap(-(v - self.mvhalf), 6.0)
        beta = 0.124 * vtrap((v - self.mvhalf), 6.0)
        return alpha, beta

    def h_gate(self, v):
        alpha = 0.015 * vtrap(v - self.hvhalf, 6.0)
        beta = 0.015 * vtrap(-(v - self.hvhalf), 6.0)
        return alpha, beta

    def xinf(self, t, u, args=None):
        v = u["v"]
        am, bm = self.m_gate(v)
        ah, bh = self.h_gate(v)
        return {"nata_m": am / (am + bm), "nata_h": ah / (ah + bh)}

    def tau(self, t, u, args=None):
        v = u["v"]
        am, bm = self.m_gate(v)
        ah, bh = self.h_gate(v)
        return {
            "nata_m": 1.0 / (am + bm) / self.tadj,
            "nata_h": 1.0 / (ah + bh) / self.tadj,
        }

    def i(self, t, u, args=None):
        v, m, h = u["v"], u["nata_m"], u["nata_h"]
        return self.gbar * m**3 * h * (v - self.ena)

    def __call__(self, t, u, args=None):
        v, m, h = u["v"], u["nata_m"], u["nata_h"]
        am, bm = self.m_gate(v)
        ah, bh = self.h_gate(v)
        dm_dt = self.tadj * (am * (1.0 - m) - bm * m)
        dh_dt = self.tadj * (ah * (1.0 - h) - bh * h)
        return {f"i_{self.name}": self.i(t, u, args), "nata_m": dm_dt, "nata_h": dh_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"nata_m": xinf["nata_m"], "nata_h": xinf["nata_h"]}
        return {"nata_m": 0.1, "nata_h": 0.1}


class Allen_NaTs(Channel):
    """Na+ channel (soma), Colbert & Pan 2002. Same as NaTa with mvhalf=-32, hvhalf=-60. States: nats_m, nats_h."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ena: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)
    mvhalf: jnp.ndarray = eqx.field(converter=jnp.array)
    hvhalf: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(
        self,
        gbar=0.01,
        ena=50.0,
        tadj=1.0,
        mvhalf=-32.0,
        hvhalf=-60.0,
    ):
        super().__init__()
        self.gbar = gbar
        self.ena = ena
        self.tadj = tadj
        self.mvhalf = mvhalf
        self.hvhalf = hvhalf

    def m_gate(self, v):
        # alpha = 0.182 * vtrap(-(v - self.mvhalf), self.mk) # allen
        # beta = 0.124 * vtrap((v - self.mvhalf), self.mk) # allen
        alpha = 0.182 * vtrap(-(v - self.mvhalf + 1e-6), 6.0)  # jaxley
        beta = 0.124 * vtrap((v - self.mvhalf + 1e-6), 6.0)  # jaxley
        return alpha, beta

    def h_gate(self, v):
        # alpha = 0.015 * vtrap(v - self.hvhalf, self.hk) # allen
        # beta = 0.015 * vtrap(-(v - self.hvhalf), self.hk) # allen
        alpha = 0.015 * vtrap(v - self.hvhalf + 1e-6, 6.0)  # jaxley
        beta = 0.015 * vtrap(-(v - self.hvhalf + 1e-6), 6.0)  # jaxley
        return alpha, beta

    def xinf(self, t, u, args=None):
        v = u["v"]
        am, bm = self.m_gate(v)
        ah, bh = self.h_gate(v)
        return {"nats_m": am / (am + bm), "nats_h": ah / (ah + bh)}

    def tau(self, t, u, args=None):
        v = u["v"]
        am, bm = self.m_gate(v)
        ah, bh = self.h_gate(v)
        return {
            "nats_m": 1.0 / (am + bm) / self.tadj,
            "nats_h": 1.0 / (ah + bh) / self.tadj,
        }

    def i(self, t, u, args=None):
        v, m, h = u["v"], u["nats_m"], u["nats_h"]
        return self.gbar * m**3 * h * (v - self.ena)

    def __call__(self, t, u, args=None):
        v, m, h = u["v"], u["nats_m"], u["nats_h"]
        am, bm = self.m_gate(v)
        ah, bh = self.h_gate(v)
        dm_dt = self.tadj * (am * (1.0 - m) - bm * m)
        dh_dt = self.tadj * (ah * (1.0 - h) - bh * h)
        return {f"i_{self.name}": self.i(t, u, args), "nats_m": dm_dt, "nats_h": dh_dt}

    def init(self, t, u, args=None):
        if "v" in u:
            xinf = self.xinf(t, u, args=args)
            return {"nats_m": xinf["nats_m"], "nats_h": xinf["nats_h"]}
        return {"nats_m": 0.1, "nats_h": 0.1}


class Allen_SK(Channel):
    """SK Ca-activated K+ channel, Kohler et al. 1996. State: sk_z. Requires u['cai'] (mM). zInf = 1/(1+(0.00043/cai)^4.8)."""

    gbar: jnp.ndarray = eqx.field(converter=jnp.array)
    ek: jnp.ndarray = eqx.field(converter=jnp.array)
    z_tau: jnp.ndarray = eqx.field(converter=jnp.array)
    tadj: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(self, gbar=0.001, ek=-77.0, z_tau=1.0, tadj=1.0):
        super().__init__()
        self.gbar = gbar
        self.ek = ek
        self.z_tau = z_tau
        self.tadj = tadj

    def z_inf(self, cai):
        cai_safe = jnp.maximum(cai, 1e-7)
        return 1.0 / (1.0 + (0.00043 / cai_safe) ** 4.8)

    def xinf(self, t, u, args=None):
        cai = u.get("cai", 1e-4)
        return {"sk_z": self.z_inf(cai)}

    def tau(self, t, u, args=None):
        return {"sk_z": self.z_tau}

    def i(self, t, u, args=None):
        v, z = u["v"], u["sk_z"]
        return self.gbar * z * (v - self.ek)

    def __call__(self, t, u, args=None):
        v, z = u["v"], u["sk_z"]
        cai = u.get("cai", 1e-4)
        z_inf = self.z_inf(cai)
        dz = (z_inf - z) / self.z_tau
        return {f"i_{self.name}": self.i(t, u, args), "sk_z": dz}

    def init(self, t, u, args=None):
        cai = u.get("cai", 1e-4)
        return {"sk_z": self.z_inf(cai)}


# Nernst reversal constants (from jaxley CaNernstReversal)
_R_GAS = 8.314  # J/(mol K)
_FARADAY = 96485.3329  # C/mol


class Allen_CaDynamics(Channel):
    """Ca2+ concentration dynamics (pump/buffer), Destexhe et al. 1994.
    Reads ica (total Ca current) from args; contributes no current, only dcai/dt.
    HH base passes required current groups via channel.requires; this channel needs total calcium current."""

    gamma: jnp.ndarray = eqx.field(converter=jnp.array)
    decay: jnp.ndarray = eqx.field(converter=jnp.array)
    depth: jnp.ndarray = eqx.field(converter=jnp.array)
    min_cai: jnp.ndarray = eqx.field(converter=jnp.array)

    requires = ("ca",)

    def __init__(self, gamma=0.05, decay=80.0, depth=0.1, min_cai=1e-4):
        super().__init__()
        self.gamma = gamma
        self.decay = decay
        self.depth = depth
        self.min_cai = min_cai

    def i(self, t, u, args=None):
        return 0.0

    def dcai_dt(self, ica, cai):
        # Allen mod equation expects ica in mA/cm^2; here currents are in uA/cm^2.
        # Equivalent conversion: -10000 * (ica_mA * ...) = -10 * (ica_uA * ...).
        flux = ica * self.gamma / (2.0 * _FARADAY * self.depth)
        return -10.0 * flux - (cai - self.min_cai) / self.decay

    def xinf(self, t, u, args=None):
        return {"cai": self.min_cai}

    def tau(self, t, u, args=None):
        return {"cai": self.decay}

    def __call__(self, t, u, args=None):
        if args is None:
            return {}  # dummy return
        ica = args[0]
        cai = u["cai"]
        dcai = self.dcai_dt(ica, cai)
        return {f"i_{self.name}": self.i(t, u, args), "cai": dcai}

    def init(self, t, u, args=None):
        return {"cai": self.min_cai}


class Allen_NernstReversal(Channel):
    """Compute Ca²⁺ reversal potential from inner and outer concentration (Nernst).
    Contributes no current; updates state 'eca' so Ca channels can use u['eca'].
    From jaxley_mech CaNernstReversal (l5pc.py)."""

    R: jnp.ndarray = eqx.field(converter=jnp.array)
    T: jnp.ndarray = eqx.field(converter=jnp.array)
    F: jnp.ndarray = eqx.field(converter=jnp.array)
    Cao: jnp.ndarray = eqx.field(converter=jnp.array)
    tau_eca: jnp.ndarray = eqx.field(
        converter=jnp.array
    )  # ms; must be << tau_cai (~900ms) for accuracy, >> 1/max|J_ode| (~1e-8ms) for stability

    def __init__(self, R=8.314, T=307.15, F=96485.3329, Cao=2.0, tau_eca=1.0):
        super().__init__()
        self.R = R
        self.T = T
        self.F = F
        self.Cao = Cao
        self.tau_eca = tau_eca

    def eca(self, u):
        """Nernst reversal eCa = (R*T/(2*F))*1000 * ln(Cao/Cai) in mV."""
        cai = jnp.maximum(u.get("cai", 1e-4), 1e-10)
        C = self.R * self.T / (2.0 * self.F) * 1000.0  # mV
        return C * jnp.log(self.Cao / cai)

    def i(self, t, u, args=None):
        return 0.0

    def xinf(self, t, u, args=None):
        return {"eca": self.eca(u)}

    def tau(self, t, u, args=None):
        return {"eca": self.tau_eca}

    def precompute(self, t, u, args=None):
        """Algebraically update eca from cai before channel evaluation."""
        return {"eca": self.eca(u)}

    def __call__(self, t, u, args=None):
        # eca is updated algebraically via precompute; freeze eca in ODE state
        return {f"i_{self.name}": self.i(t, u, args), "eca": jnp.zeros_like(u.get("eca", jnp.array(0.0)))}

    def init(self, t, u, args=None):
        cai = u.get("cai", 1e-4)
        u_tmp = {**u, "cai": cai}
        return {"eca": self.eca(u_tmp)}


class OmniGate(eqx.Module):
    state_name: str = eqx.field(static=True)

    vh: jnp.ndarray = eqx.field(converter=jnp.array)
    A: jnp.ndarray = eqx.field(converter=jnp.array)

    a: jnp.ndarray = eqx.field(converter=jnp.array)
    b: jnp.ndarray = eqx.field(converter=jnp.array)

    b1: jnp.ndarray = eqx.field(converter=jnp.array)
    c1: jnp.ndarray = eqx.field(converter=jnp.array)
    d1: jnp.ndarray = eqx.field(converter=jnp.array)
    e1: jnp.ndarray = eqx.field(converter=jnp.array)

    b2: jnp.ndarray = eqx.field(converter=jnp.array)
    c2: jnp.ndarray = eqx.field(converter=jnp.array)
    d2: jnp.ndarray = eqx.field(converter=jnp.array)
    e2: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(
        self,
        state_name,
        vh=0.0,
        A=10.0,
        a=0.0,
        b=0.0,
        b1=0.0,
        c1=0.0,
        d1=0.0,
        e1=0.0,
        b2=0.0,
        c2=0.0,
        d2=0.0,
        e2=0.0,
    ):
        self.state_name = state_name

        self.a = a
        self.b = b

        self.b1 = b1
        self.c1 = c1
        self.d1 = d1
        self.e1 = e1

        self.b2 = b2
        self.c2 = c2
        self.d2 = d2
        self.e2 = e2

        self.vh = vh
        self.A = A

    @property
    def k1(self):
        return jnp.array([self.b1, self.c1, self.d1, self.e1])

    @property
    def k2(self):
        return jnp.array([self.b2, self.c2, self.d2, self.e2])

    def xinf(self, t, u, args=None):
        v = u["v"]
        u_inf = 1.0 / (1.0 + safe_exp((self.b - self.a * v)))
        return {self.state_name: u_inf}

    def B(self, v):
        _v = v - self.vh
        p = self.k1 * _v ** jnp.array([1.0, 2.0, 3.0, 4.0])
        return safe_exp(-jnp.sum(p))

    def C(self, v):
        _v = v - self.vh
        p = self.k2 * _v ** jnp.array([1.0, 2.0, 3.0, 4.0])
        return safe_exp(jnp.sum(p))

    def tau(self, t, u, args=None):
        v = u["v"]
        tau = self.A / (self.B(v) + self.C(v))
        safe_tau = jnp.clip(tau, 1e-3, 1e4)
        return {self.state_name: safe_tau}

    def __call__(self, t, u, args=None):
        u_gate = u[self.state_name]
        xinf = self.xinf(t, u, args)[self.state_name]
        tau = self.tau(t, u, args)[self.state_name]

        du_dt = (xinf - u_gate) / tau
        return {self.state_name: du_dt}


class Omni(Channel):
    """Omni channel model from Chintaluri et al.
    https://www.biorxiv.org/content/10.1101/2025.10.03.680368v1.full.pdf
    """

    gx: jnp.ndarray = eqx.field(converter=jnp.array)
    ex: jnp.ndarray = eqx.field(converter=jnp.array)
    powx: dict[str, jnp.ndarray]
    gates: dict[str, OmniGate] = eqx.field(converter=dict)

    def __init__(
        self,
        latent_states=None,
        key=None,
    ):
        super().__init__()
        self.gx = jnp.array(100.0)
        self.ex = jnp.array(50.0)

        if latent_states is None:
            self.gates = {}
            self.powx = {}
        else:
            self.gates = {k: OmniGate(k) for k in latent_states}
            self.powx = {k: jnp.array(1.0) for k in latent_states}

    def _compute_gates(self, t, u, args=None):
        u_inf = {}
        tau = {}
        for k, g in self.gates.items():
            u_inf[k] = g.xinf(t, u, args)[k]
            tau[k] = g.tau(t, u, args)[k]
        return u_inf, tau

    def i(self, t, u, args=None):
        u_no_v = {k: u[k] for k in self.gates.keys()}
        log_pow = lambda pow, x: pow * safe_log(x)
        log_latents = jax.tree.map(log_pow, self.powx, u_no_v)
        gates = safe_exp(jax.tree_util.tree_reduce(lambda x, y: x + y, log_latents))

        i_ion = gates * self.gx * (u["v"] - self.ex)
        return i_ion

    def __call__(self, t, u, args=None):
        du = {f"i_{self.name}": self.i(t, u, args)}
        for k, g in self.gates.items():
            du.update(g(t, u, args))
        return du

    def init(self, t, u, args=None):
        if "v" in u:
            u0 = {"v": u["v"]}
            for k, g in self.gates.items():
                u0.update(g.xinf(t, u, args))
            return u0
        return {**{k: 0.1 for k in self.gates.keys()}, "v": -70.0}


class MLP(eqx.nn.MLP):
    skip_connections: bool

    def __init__(
        self,
        *args,
        layer_norm=True,
        last_layer_initializer=None,
        initializer=None,
        key: jax.Array = jr.key(0),
        skip_connections=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, key=key)
        orig_layers = list(self.layers)
        layers = []
        self.skip_connections = skip_connections

        initializer = (
            jax.nn.initializers.uniform() if initializer is None else initializer
        )
        last_layer_initializer = (
            initializer if last_layer_initializer is None else last_layer_initializer
        )

        for i, layer in enumerate(orig_layers):
            # Only apply init to Linear layers
            is_linear = isinstance(layer, eqx.nn.Linear)
            if is_linear:
                # Make a new rng key for each layer
                key, subkey = jr.split(key, 2)
                if i == len(orig_layers) - 1:
                    new_weight = last_layer_initializer(subkey, layer.weight.shape)
                else:
                    new_weight = initializer(subkey, layer.weight.shape)
                new_bias = (
                    jnp.zeros_like(layer.bias) if layer.bias is not None else None
                )
                layer = eqx.tree_at(
                    lambda l: (l.weight, l.bias), layer, replace=(new_weight, new_bias)
                )
            layers.append(layer)

            if layer_norm and is_linear and i < len(orig_layers) - 1:
                layers.append(eqx.nn.LayerNorm(layer.out_features))
        self.layers = tuple(layers)

    def __call__(self, x, *, key: jax.Array | None = None):
        """Forward pass. When skip_connections is True, adds ResNet-style skip
        connections (x + F(x)) between hidden blocks of matching dimension.
        """
        if not self.skip_connections:
            # Match parent eqx.nn.MLP: iterate layers, apply activation after
            # each hidden block (Linear->LayerNorm), final_activation after last Linear
            block_idx = 0
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if isinstance(layer, eqx.nn.LayerNorm):
                    layer_activation = jax.tree_util.tree_map(
                        lambda a, idx=block_idx: a[idx] if eqx.is_array(a) else a,
                        self.activation,
                    )
                    x = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, x)
                    block_idx += 1
                elif i == len(self.layers) - 1:
                    if self.out_size == "scalar":
                        x = self.final_activation(x)
                    else:
                        x = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, x)
            return x

        # ResNet-style skip: x_skip + block(x) for hidden blocks with same in/out size
        block_idx = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, eqx.nn.Linear):
                if block_idx >= 1 and block_idx < self.depth:
                    x_skip = x
                x = layer(x)
                if i == len(self.layers) - 1:
                    if self.out_size == "scalar":
                        x = self.final_activation(x)
                    else:
                        x = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, x)
            else:
                x = layer(x)
                layer_activation = jax.tree_util.tree_map(
                    lambda a, idx=block_idx: a[idx] if eqx.is_array(a) else a,
                    self.activation,
                )
                x = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, x)
                if block_idx >= 1 and block_idx < self.depth:
                    x = x + x_skip
                block_idx += 1
        return x


class EnsembleNet(eqx.Module):
    ensemble: MLP
    in_map: jnp.ndarray
    out_map: jnp.ndarray
    n_nets: int
    max_in: int
    max_out: int

    def __init__(
        self,
        input_mapping: list[list[int]],
        output_sizes: list[int],
        width_size: int = 16,
        depth: int = 2,
        activation=jax.nn.relu,
        final_activation=lambda x: x,
        layer_norm=True,
        last_layer_initializer=None,
        initializer=None,
        skip_connections=False,
        key: jr.PRNGKey = jr.key(0),
    ):
        self.n_nets = len(input_mapping)
        self.max_in = max(len(idx) for idx in input_mapping)
        self.max_out = max(output_sizes)

        n_features = max(max(idx) for idx in input_mapping) + 1

        # Build input mapping matrix M
        M = jnp.zeros((self.n_nets * self.max_in, n_features))
        for i, idx_list in enumerate(input_mapping):
            for j, idx in enumerate(idx_list):
                M = M.at[i * self.max_in + j, idx].set(1.0)
        self.in_map = M

        # Build output unmapping matrix N
        total_out = sum(output_sizes)
        N = jnp.zeros((total_out, self.n_nets * self.max_out))
        row = 0
        for i, size in enumerate(output_sizes):
            for j in range(size):
                N = N.at[row, i * self.max_out + j].set(1.0)
                row += 1
        self.out_map = N

        # Create ensemble
        @eqx.filter_vmap
        def make_ensemble(k):
            return MLP(
                self.max_in,
                self.max_out,
                width_size,
                depth,
                activation=activation,
                final_activation=final_activation,
                key=k,
                layer_norm=layer_norm,
                last_layer_initializer=last_layer_initializer,
                initializer=initializer,
                skip_connections=skip_connections,
            )

        self.ensemble = make_ensemble(jr.split(key, self.n_nets))

    def __call__(self, x):
        # x → M@x → reshape → ensemble → reshape → N@y → output
        x_mapped = (self.in_map @ x).reshape(self.n_nets, self.max_in)

        @eqx.filter_vmap
        def forward(model, inp):
            return model(inp)

        y_padded = forward(self.ensemble, x_mapped).reshape(-1)
        return self.out_map @ y_padded


class StackedNet(eqx.Module):
    nets: Union[MLP, EnsembleNet]
    num_nets: int = eqx.field(static=True)

    def __init__(
        self,
        net: Union[MLP, EnsembleNet],
        key: jr.PRNGKey,
        num_nets: int,
    ):
        self.num_nets = num_nets

        @eqx.filter_vmap
        def make_nets(k):
            return self._reinit_like(net, k)

        self.nets = make_nets(jr.split(key, num_nets))

    @staticmethod
    def _reinit_like(net: Union[MLP, EnsembleNet], key: jr.PRNGKey):
        if isinstance(net, MLP):
            has_layer_norm = any(
                isinstance(layer, eqx.nn.LayerNorm) for layer in net.layers
            )
            return MLP(
                net.in_size,
                net.out_size,
                net.width_size,
                net.depth,
                activation=net.activation,
                final_activation=net.final_activation,
                use_bias=net.use_bias,
                layer_norm=has_layer_norm,
                skip_connections=net.skip_connections,
                key=key,
            )

        if isinstance(net, EnsembleNet):
            first = jax.tree.map(
                lambda leaf: (
                    leaf[0]
                    if eqx.is_array(leaf)
                    and leaf.ndim > 0
                    and leaf.shape[0] == net.n_nets
                    else leaf
                ),
                net.ensemble,
            )
            has_layer_norm = any(
                isinstance(layer, eqx.nn.LayerNorm) for layer in first.layers
            )

            input_mapping = []
            for i in range(net.n_nets):
                idxs = []
                for j in range(net.max_in):
                    row = net.in_map[i * net.max_in + j]
                    if jnp.any(row):
                        idxs.append(int(jnp.argmax(row)))
                input_mapping.append(idxs)

            output_sizes = []
            for i in range(net.n_nets):
                cols = net.out_map[:, i * net.max_out : (i + 1) * net.max_out]
                output_sizes.append(int(jnp.sum(jnp.any(cols > 0, axis=1))))

            return EnsembleNet(
                input_mapping=input_mapping,
                output_sizes=output_sizes,
                width_size=first.width_size,
                depth=first.depth,
                activation=first.activation,
                final_activation=first.final_activation,
                layer_norm=has_layer_norm,
                skip_connections=first.skip_connections,
                key=key,
            )

        raise TypeError("StackedNet only supports MLP or EnsembleNet instances.")

    def __call__(self, x):
        @eqx.filter_vmap
        def forward(model):
            return model(x)

        return forward(self.nets)


class ZScoreLayer(eqx.Module):
    mean: dict[str, jnp.ndarray]
    std: dict[str, jnp.ndarray]

    def __init__(self, state_mean, state_std):
        self.mean = {k: 0.0 if v is None else v for k, v in state_mean.items()}
        self.std = {k: 1.0 if v is None else v for k, v in state_std.items()}

    def __call__(self, x):
        fw = lambda x, mean, std: jnp.clip((x - mean) / std, -1e2, 1e2)
        return jax.tree.map(fw, x, self.mean, self.std)

    def inverse(self, x):
        inv = lambda x, mean, std: jnp.clip(x, -1e2, 1e2) * std + mean
        return jax.tree.map(inv, x, self.mean, self.std)


class NODE(Channel):
    net: Union[MLP, EnsembleNet, StackedNet]
    ipt_tf: Union[ZScoreLayer, callable]
    out_tf: Union[ZScoreLayer, callable]
    latent_states: list[str] = eqx.field(static=True)
    restrict_state: bool = eqx.field(static=True)
    incl_time: bool = eqx.field(static=True)
    i_affine: dict[str, jnp.ndarray] = eqx.field(converter=dict)
    is_split: bool = eqx.field(static=True)
    return_current: bool = eqx.field(static=True)

    def __init__(
        self,
        latent_states=None,
        width_size=32,
        depth_size=1,
        activation=jax.nn.softplus,
        last_layer_initializer=None,
        share_weights=False,
        split_model=False,
        initializer=jax.nn.initializers.normal(),
        skip_connections=False,
        restrict_state=False,
        incl_time=False,
        use_layer_norm=True,
        return_current=True,
        *,
        key: jax.Array = jr.key(0),
    ):
        super().__init__()
        latent_states = [] if latent_states is None else latent_states

        if share_weights:
            in_size = 1 + len(latent_states)  # v + latent states
            in_size += 1 if incl_time else 0
            out_size = 1 + len(latent_states)  # i + latent states
            self.net = MLP(
                in_size,
                out_size,
                width_size,
                depth_size,
                key=key,
                activation=activation,
                final_activation=lambda x: x,
                last_layer_initializer=last_layer_initializer,
                initializer=initializer,
                layer_norm=use_layer_norm,
            )
        else:
            # Create input mapping
            num_states = 1 if latent_states is None else len(latent_states) + 1
            v_ipts = list(range(num_states))
            v_ipts += [num_states] if incl_time else []
            
            # latents_ipts = [[0, i] for i in range(1, num_states)]
            # map_inputs = [v_ipts] + latents_ipts
            # out_sizes = [1] * num_states  # i(v, x1, x2, ...), x1(v, x1), x2(v, x2), ...
            
            map_inputs = [v_ipts] + [[i for i in range(num_states)]]*len(latent_states)  # all latents see all inputs
            out_sizes = [1] * num_states  # i(v, x1, x2, ...), x1(v, x1, x2, ...)

            # map_inputs = [v_ipts] + [[i for i in range(num_states)]]
            # out_sizes = [1, len(latent_states)]
            self.net = EnsembleNet(
                input_mapping=map_inputs,
                output_sizes=out_sizes,
                width_size=width_size,
                depth=depth_size,
                key=key,
                activation=activation,
                final_activation=lambda x: x,
                last_layer_initializer=last_layer_initializer,
                initializer=initializer,
                skip_connections=skip_connections,
                layer_norm=use_layer_norm,
            )
        if split_model:
            self.net = StackedNet(self.net, key=key, num_nets=2)

        self.latent_states = [] if latent_states is None else latent_states

        init_mean = lambda: {k: 0.0 for k in ["v", *latent_states]}
        init_std = lambda: {k: 1.0 for k in ["v", *latent_states]}
        self.ipt_tf = ZScoreLayer(init_mean(), init_std())
        self.out_tf = ZScoreLayer(init_mean(), init_std())
        self.restrict_state = restrict_state
        self.incl_time = incl_time
        self.i_affine = {"weight": jnp.array(1.0), "bias": jnp.array(0.0)}
        self.is_split = split_model
        self.return_current = return_current

    def __repr__(self):
        cls_name = self.__class__.__name__
        latent_states_str = ", ".join([f"{k}" for k in self.latent_states])
        return f"{cls_name}(latent_states=[{latent_states_str}])"

    def __call__(self, t, u, args=None):
        u_node = {k: u[k] for k in ["v"] + self.latent_states}
        # v_min, v_max = -80.0, 60.0
        # u_node["v"] = 2.0 * (u["v"] - v_min) / (v_max - v_min) - 1.0
        
        u_node = self.ipt_tf(u_node)  # stabilize and re-scale input

        u_flat = jnp.hstack([u_node[k] for k in ["v"] + self.latent_states])
        if self.incl_time:
            u_flat = jnp.concatenate([u_flat, t[None]])
        out_flat = self.net(u_flat)

        if self.is_split:
            out_a = {
                k: out_flat[0, i] for i, k in enumerate(["v"] + self.latent_states)
            }
            out_b = {
                k: out_flat[1, i] for i, k in enumerate(["v"] + self.latent_states)
            }
            du_out = jax.tree.map(lambda a, x, b: a * x + b, out_a, u_node, out_b)
        else:
            du_out = {k: out_flat[i] for i, k in enumerate(["v"] + self.latent_states)}
        # du_out = jax.tree.map(lambda x: x * jnp.maximum(self.out_scale, 1e-4), du_out)  # global scaling
        # du_out = jax.tree.map(lambda x, std: jnp.clip(x, -1e2, 1e2) * std, out, self.ipt_tf.std)  # manual inverse to avoid clipping in ZScoreLayer
        du_out = self.out_tf.inverse(du_out)  # stabilize and re-scale output

        if self.restrict_state:
            for k in self.latent_states:
                # du_out[k] = u[k] * (1 - u[k]) * du_out[k]  # restrict state to [0, 1]
                du_out[k] = du_out[k] - 0.1*u_node[k]  # restrict state to [0, 1]

        # du_out[f"i_{self.name}"] = self.i_affine["weight"] * du_out.pop("v") + self.i_affine["bias"]
        if self.return_current:
            du_out[f"i_{self.name}"] = du_out.pop("v")
        return du_out

    def init(self, t, u, args=None):
        return {k: jnp.array(0.1) for k in self.latent_states}

    def zscore(self, state_mean=None, state_std=None, deriv_mean=None, deriv_std=None):
        if deriv_mean is not None:
            warnings.warn(
                "deriv_mean != 0. Are you sure you want a shifted derivative?"
            )
        init_w = lambda x: {k: x for k in ["v", *self.latent_states]}
        init_mean = init_w(0.0)
        init_std = init_w(1.0)

        state_mean = init_mean if state_mean is None else state_mean
        missing_states = set(init_mean) - set(state_mean)
        none_states = {k for k in state_mean if state_mean[k] is None}
        for k in missing_states | none_states:
            state_mean[k] = init_mean[k]

        state_std = init_std if state_std is None else state_std
        missing_states = set(init_std) - set(state_std)
        none_states = {k for k in state_std if state_std[k] is None}
        for k in missing_states | none_states:
            state_std[k] = init_std[k]

        deriv_mean = init_mean if deriv_mean is None else deriv_mean
        missing_states = set(init_mean) - set(deriv_mean)
        none_states = {k for k in deriv_mean if deriv_mean[k] is None}
        for k in missing_states | none_states:
            deriv_mean[k] = init_mean[k]

        ipt_tf = ZScoreLayer(state_mean, state_std)

        deriv_mean = init_mean if deriv_mean is None else deriv_mean
        missing_derivs = set(init_mean) - set(deriv_mean)
        none_derivs = {k for k in deriv_mean if deriv_mean[k] is None}
        for k in missing_derivs | none_derivs:
            deriv_mean[k] = init_mean[k]

        deriv_std = init_std if deriv_std is None else deriv_std
        missing_derivs = set(init_std) - set(deriv_std)
        none_derivs = {k for k in deriv_std if deriv_std[k] is None}
        for k in missing_derivs | none_derivs:
            deriv_std[k] = init_std[k]

        out_tf = ZScoreLayer(deriv_mean, deriv_std)
        return eqx.tree_at(lambda m: (m.ipt_tf, m.out_tf), self, (ipt_tf, out_tf))

    def zscore_w_data(self, state_data=None, deriv_data=None):
        state_mean = None if state_data is None else jax.tree.map(jnp.mean, state_data)
        state_std = None if state_data is None else jax.tree.map(jnp.std, state_data)

        deriv_mean = None  # always 0.0, since deriv should not be shifted
        deriv_std = None if deriv_data is None else jax.tree.map(jnp.std, deriv_data)

        return self.zscore(state_mean, state_std, deriv_mean, deriv_std)


class BioPhysicsNODE1(NODE):
    powx: dict[str, jnp.ndarray] = eqx.field(converter=dict)
    gx: jnp.ndarray = eqx.field(converter=jnp.array)
    ex: jnp.ndarray = eqx.field(converter=jnp.array)
    # extra: dict[str, jnp.ndarray] = eqx.field(converter=dict)

    def __init__(
        self,
        latent_states=None,
        width_size=8,
        depth_size=2,
        activation=jax.nn.mish,
        last_layer_initializer=None,
        share_weights=False,
        initializer=jax.nn.initializers.normal(),
        skip_connections=False,
        use_layer_norm=True,
        *,
        key: jax.Array = jr.key(0),
    ):
        super().__init__(
            latent_states=latent_states,
            width_size=width_size,
            depth_size=depth_size,
            activation=activation,
            initializer=initializer,
            skip_connections=skip_connections,
            use_layer_norm=use_layer_norm,
            incl_time=False,
            key=key,
        )
        self.name = "node"

        if share_weights:
            in_size = 1  # v
            out_size = 2 * len(latent_states)  # (u_inf, tau) for each latent states
            self.net = MLP(
                in_size,
                out_size,
                width_size,
                depth_size,
                key=key,
                activation=activation,
                final_activation=lambda x: x,
                last_layer_initializer=last_layer_initializer,
                initializer=initializer,
                skip_connections=skip_connections,
            )
        else:
            latent_states = [] if latent_states is None else latent_states
            if len(latent_states) == 0:
                self.net = lambda x: x
            else:
                num_latents = len(latent_states)
                input_mapping = [[0]] * num_latents
                out_sizes = [2] * num_latents

                self.net = EnsembleNet(
                    input_mapping=input_mapping,
                    output_sizes=out_sizes,
                    width_size=width_size,
                    depth=depth_size,
                    key=key,
                    activation=activation,
                    final_activation=lambda x: x,
                    last_layer_initializer=last_layer_initializer,
                    initializer=initializer,
                    skip_connections=skip_connections,
                )

        self.powx = {k: jnp.array(1.0) for k in self.latent_states}
        self.gx = jnp.array(150.0)
        self.ex = jnp.array(50.0)
        # self.extra = {"weight": jnp.array([1.0]*num_latents), "bias": jnp.array([-0.01]*num_latents)}

    def zscore(self, state_mean=None, state_std=None, deriv_mean=None, deriv_std=None):
        warnings.warn("BioPhysicsNODE.zscore will ignore derivs")
        return super().zscore(state_mean, state_std, None, None)

    def i(self, t, u, args=None):
        u_no_v = {k: u[k] for k in self.latent_states}
        log_pow = lambda pow, x: pow * safe_log(x)
        log_latents = jax.tree.map(log_pow, self.powx, u_no_v)
        gates = safe_exp(jax.tree_util.tree_reduce(lambda x, y: x + y, log_latents))
        return gates * self.gx * (u["v"] - self.ex)
    
        # log_gates = jax.tree_util.tree_reduce(lambda x, y: x + y, log_latents)
        # i_ion = self.gx * safe_exp(log_gates) * (u["v"] - self.ex)
        # return i_ion

    def _compute_gates(self, t, u, args=None):
        # mean = self.ipt_tf.mean["v"]
        # std = self.ipt_tf.std["v"]
        # u_in = jnp.atleast_1d((u["v"] - mean) / std)
        
        v_min, v_max = -80.0, 60.0
        u_in = 2.0 * (u["v"] - v_min) / (v_max - v_min) - 1.0
        net_out = self.net(jnp.atleast_1d(u_in))

        gate_params = net_out.reshape(-1, 2)  # [u_inf, tau] pairs
        u_inf = jax.nn.sigmoid(gate_params[:, 0])  # Steady-state in [0, 1]

        tau_safe = safe_exp(gate_params[:, 1])
        # tau_safe = 1e-3 + jax.nn.softplus(gate_params[:, 1]) # prev

        # # --- tau: shape-scale decomposition ---
        # shape = jax.nn.sigmoid(gate_params[:, 1])
        # lb = self.extra["bias"]  # lower bound in log-space
        # ub = lb + jax.nn.softplus(self.extra["weight"])
        # log_tau = lb + shape * (ub - lb)  # scale to [lb, ub]
        # tau_safe = safe_exp(log_tau)

        tau = {s: tau_safe[i] for i, s in enumerate(self.latent_states)}
        u_inf = {s: u_inf[i] for i, s in enumerate(self.latent_states)}

        return u_inf, tau

    def __call__(self, t, u, args=None):
        u_no_v = {k: u[k] for k in self.latent_states}

        if len(self.latent_states) != 0:
            u_inf, tau = self._compute_gates(t, u, args)
            clip = lambda x: jnp.clip(x, 1e-10, 1.0 - 1e-10)
            dgate = {k: (u_inf[k] - clip(u[k])) / tau[k] for k in self.latent_states}

            log_pow = lambda pow, x: pow * safe_log(x)
            log_latents = jax.tree.map(log_pow, self.powx, u_no_v)
            gates = safe_exp(jax.tree_util.tree_reduce(lambda x, y: x + y, log_latents))

            # powx = lambda pow, x: x ** pow
            # latents = jax.tree.map(powx, self.powx, u_no_v)
            # gates = jax.tree_util.tree_reduce(lambda x, y: x * y, latents)
        else:
            gates = jnp.array(1.0)
            dgate = {}

        i_ion = gates * self.gx * (u["v"] - self.ex)
        du_out = {f"i_{self.name}": i_ion, **dgate}
        return du_out

    def init(self, t, u, args=None):
        if "v" in u:
            u0 = {"v": u["v"]}
            if len(self.latent_states) != 0:
                u_init = {k: u.get(k, jnp.array(0.0)) for k in ["v"] + self.latent_states}
                u_inf, _ = self._compute_gates(t, u_init, args)
                u0.update(u_inf)
            return u0
        return {**{k: 0.1 for k in self.latent_states}, "v": -70.0}


class BioPhysicsNODE2(NODE):
    ex: jnp.ndarray = eqx.field(converter=jnp.array)

    def __init__(
        self,
        latent_states=None,
        width_size=8,
        depth_size=2,
        activation=jax.nn.tanh,
        last_layer_initializer=None,
        share_weights=False,
        initializer=jax.nn.initializers.normal(),
        skip_connections=False,
        use_layer_norm=True,
        incl_time=False,
        *,
        key: jax.Array = jr.key(0),
    ):
        super().__init__(
            latent_states=latent_states,
            width_size=width_size,
            depth_size=depth_size,
            activation=activation,
            initializer=initializer,
            skip_connections=skip_connections,
            use_layer_norm=use_layer_norm,
            incl_time=incl_time,
            key=key,
        )
        self.name = "node"

        latent_states = [] if latent_states is None else latent_states
        num_latents = len(latent_states)

        if share_weights:
            in_size = 1 + num_latents  # v + latent states
            in_size += 1 if incl_time else 0
            out_size = (
                1 + 2 * num_latents
            )  # gating + (u_inf, tau) for each latent state
            self.net = MLP(
                in_size,
                out_size,
                width_size,
                depth_size,
                key=key,
                activation=activation,
                final_activation=lambda x: x,
                last_layer_initializer=last_layer_initializer,
                initializer=initializer,
                skip_connections=skip_connections,
            )
        else:
            num_0_ipts = num_latents + 1
            num_0_ipts += 1 if incl_time else 0
            input_mapping = [list(range(num_0_ipts))] + [
                [0, i + 1] for i in range(num_latents)
            ]
            out_sizes = [1] + [2] * num_latents

            self.net = EnsembleNet(
                input_mapping=input_mapping,
                output_sizes=out_sizes,
                width_size=width_size,
                depth=depth_size,
                key=key,
                activation=activation,
                final_activation=lambda x: x,
                last_layer_initializer=last_layer_initializer,
                initializer=initializer,
            )

        self.ex = jnp.array(50.0)
        self.incl_time = incl_time

    def zscore(self, state_mean=None, state_std=None, deriv_mean=None, deriv_std=None):
        warnings.warn("BioPhysicsNODE.zscore will ignore derivs")
        return super().zscore(state_mean, state_std, None, None)

    def i(self, t, u, args=None):
        _, gating_factor = self._compute_gates(t, u, args)
        return gating_factor * (u["v"] - self.ex)

    def _compute_gates(self, t, u, args=None):
        u_in = {k: u[k] for k in ["v"] + self.latent_states}
        u_tf = self.ipt_tf(u_in)  # stabilize and re-scale input
        u_flat = jnp.hstack([u_tf["v"], *[u_tf[k] for k in self.latent_states]])
        if self.incl_time:
            u_flat = jnp.concatenate([u_flat, t[None]])
        net_out = self.net(u_flat)

        gating_factor = jax.nn.softplus(net_out[0])
        gate_params = net_out[1:].reshape(-1, 2)  # [u_inf, tau] pairs

        u_inf = jax.nn.sigmoid(gate_params[:, 0])  # Steady-state in [0, 1]
        tau_safe = safe_exp(gate_params[:, 1])

        tau = {s: tau_safe[i] for i, s in enumerate(self.latent_states)}
        u_inf = {s: u_inf[i] for i, s in enumerate(self.latent_states)}

        return u_inf, tau, gating_factor

    def __call__(self, t, u, args=None):
        u_inf, tau, gating_factor = self._compute_gates(t, u, args)
        dgate = {k: (u_inf[k] - u[k]) / tau[k] for k in self.latent_states}

        i_ion = gating_factor * (u["v"] - self.ex)
        du_out = {f"i_{self.name}": i_ion, **dgate}
        return du_out

    def init(self, t, u, args=None):
        if "v" in u:
            u0 = {"v": u["v"]}
            if len(self.latent_states) != 0:
                u_init = {k: u.get(k, jnp.array(0.0)) for k in ["v"] + self.latent_states}
                u_inf, _, _ = self._compute_gates(t, u_init, args)
                u0.update(u_inf)
            return u0
        return {**{k: 0.1 for k in self.latent_states}, "v": -70.0}
