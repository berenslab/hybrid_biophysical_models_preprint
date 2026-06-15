import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap
import diffrax
import jax
from hybrid_models.utils import label_params, filter_params_by_label


def plot_cascade_results(ts, ys, ys_pred=None, stim=None, ax=None, title=None):
    # Plot results
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(ts, ys[:, 0], "b-", label="x1(t) - True")
    ax.plot(ts, ys[:, 1], "r-", label="x2(t) - True")
    if stim:
        ax.plot(ts, stim(ts), "g-", label="Input u(t)")
    if ys_pred is not None:
        ax.plot(ts, ys_pred[:, 0], "b--", label="x1(t) - Predicted")
        ax.plot(ts, ys_pred[:, 1], "r--", label="x2(t) - Predicted")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    title = "Response of Cascaded System to Step Input" if title is None else title
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return ax


def plot_loss(losses, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training Loss")
    ax.grid(True)
    return ax


def plot_voltage_and_current(
    model,
    ts,
    initial_state,
    current_name=None,
    observed_dims=(0,),
    axs=None,
    **kwargs,
):
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(5, 2), layout="constrained")

    def integrate(func, ts, y0, **kwargs):
        """Generate training data using the true system"""

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            diffrax.Tsit5(),
            # diffrax.Kvaerno3(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            # stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6), # makes things faster but less stable
            stepsize_controller=diffrax.PIDController(
                rtol=1e-10, atol=1e-12
            ),  # makes things more stable but slower
            saveat=diffrax.SaveAt(ts=ts),
            **kwargs,
        )
        return solution.ts, solution.ys

    ts, xs = integrate(model, ts, initial_state)

    ys = xs[:, observed_dims]
    axs[0].plot(ts, ys, **kwargs)
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Voltage (mV)")
    axs[0].set_title("Voltage")

    if current_name is not None:
        i_func = lambda u: vmap(model.channels[current_name], in_axes=(None, 0, None))(
            0.0, u, None
        )[:, 0]
        state_inds = model.state_inds[current_name]
        u_t = xs[:, state_inds]
        i_t = i_func(u_t)

        axs[1].plot(ts, i_t, **kwargs)
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Current (nA)")
        axs[1].set_title("Predicted I(t)")

        if len(state_inds) == 1 and state_inds[0] == 0:
            vs = jnp.linspace(-100, 50, 100).reshape(-1, 1)
            i_v = i_func(vs)
            if u_t.shape[1] == 1:  # only depends on voltage
                axs[2].plot(vs, i_v, **kwargs)
                axs[2].set_xlabel("Voltage (mV)")
                axs[2].set_ylabel("Current (nA)")
                axs[2].set_title("Predicted I(V)")
        else:
            axs[2].plot(ts, u_t[:, 1:], **kwargs)
            axs[2].set_xlabel("Time (ms)")
            axs[2].set_ylabel("Latent state")
    return axs


def plot_gradient_norms(
    grad_pytrees, log_scale: bool = True, seperate_labels=["ode", "nn"], **kwargs
):
    total_norms = {k: [] for k in seperate_labels}

    for grad_tree in grad_pytrees:
        for target in seperate_labels:
            if target in ["ode", "nn"]:
                labels = label_params(
                    grad_tree, map_if_path_contains={"mlp": "nn"}, otherwise="ode"
                )
            else:
                labels = label_params(grad_tree, {target: target}, "")
            grads = filter_params_by_label(grad_tree, labels, target)

            grad_flat, _ = jax.tree_util.tree_flatten(grads)
            grad_flat = [g for g in grad_flat if g is not None]

            if grad_flat:
                total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_flat))
                total_norms[target].append(float(total_norm))
            else:
                total_norms[target].append(0.0)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for label in seperate_labels:
        ax.plot(total_norms[label], **kwargs, label=label)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Gradient Norm")
    ax.set_title("Gradient Norm Over Training")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if log_scale:
        ax.set_yscale("log")
    return ax
