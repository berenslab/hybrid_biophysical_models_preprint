import jax
import jax.numpy as jnp
import jax.random as jr
from numpy import dtype
import optax
import equinox as eqx

from hybrid_models.utils import label_params


def scale_by_param_magnitude(epsilon=1e-8):
    """
    Scale gradients by 1/sqrt(|param|) to normalize impact across different parameter scales.
    This helps balance parameters that live on different orders of magnitude.

    Args:
        epsilon: Small value to avoid division by zero (default: 1e-8)
    """

    def init_fn(params):
        return {}

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state

        def scale_fn(g, p):
            param_mag = jnp.abs(p) + epsilon
            return g / jnp.sqrt(param_mag)

        scaled_updates = jax.tree.map(scale_fn, updates, params)
        return scaled_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_transform_magnitude(epsilon=1e-8, model_param_transform=None):
    """
    Remove the effect of parameter transforms on gradients.

    When parameters are stored in unconstrained space and transforms (e.g., sigmoid/logistic)
    are applied in the loss function, the gradient w.r.t. unconstrained parameters includes
    the transform Jacobian: grad_unconstrained = grad_constrained × d(transform)/d(p).

    This function scales gradients by 1/|Jacobian| to cancel out the transform's effect,
    allowing optimization to proceed as if parameters were not transformed. This removes
    the dependency on where the parameter lies on the sigmoid curve.

    Each parameter uses its own transform with its specific bounds, so the Jacobian
    is computed correctly for each parameter's transform.

    Args:
        epsilon: Small value to avoid division by zero (default: 1e-8)
        model_param_transform: ModelParamTransform object. Scales gradients by 1/|Jacobian|
            to remove transform effects, using the appropriate transform for each parameter
            based on its path.
    """

    def init_fn(params):
        return {}

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state

        if model_param_transform is None:
            return updates, state

        # Get transform dictionary mapping parameter names to transforms
        tf_dict = model_param_transform.param_tf.tf_dict

        # Build path-to-transform mapping by matching parameter names in paths
        path_to_transform = {}

        def collect_transforms(path, value):
            if eqx.is_array(value):
                path_str = "".join([str(p) for p in path])
                # Match transform by parameter name in path
                for param_name, transform in tf_dict.items():
                    if param_name in path_str:
                        path_to_transform[path_str] = transform
                        break
            return value

        jax.tree_util.tree_map_with_path(
            collect_transforms, params, is_leaf=eqx.is_array
        )

        def scale_fn_with_path(path, g, p):
            path_str = "".join([str(p) for p in path])
            transform = path_to_transform.get(path_str)

            if transform is not None:
                # Compute transform Jacobian: d(transform)/d(p_unconstrained)
                # For scalars, use grad directly; for arrays, use vmap(grad)
                if p.ndim == 0:
                    jacobian = jax.grad(transform.forward)(p)
                else:
                    jacobian = jax.vmap(jax.grad(transform.forward))(p)

                # Scale by 1/|Jacobian| to cancel the transform's effect on gradients
                # This makes optimization proceed as if parameters were not transformed
                jacobian_mag = jnp.abs(jacobian) + epsilon
                scale = 1.0 / jacobian_mag
            else:
                # No transform found: no scaling needed
                scale = 1.0

            return g * scale

        # Apply scaling with path information
        updates_leaves, updates_treedef = jax.tree_util.tree_flatten(updates)
        params_leaves, _ = jax.tree_util.tree_flatten(params)
        paths_list = []

        def collect_paths(path, value):
            if eqx.is_array(value):
                paths_list.append(path)
            return value

        jax.tree_util.tree_map_with_path(collect_paths, updates, is_leaf=eqx.is_array)

        scaled_leaves = [
            scale_fn_with_path(path, g, p)
            for path, g, p in zip(paths_list, updates_leaves, params_leaves)
        ]
        scaled_updates = jax.tree_util.tree_unflatten(updates_treedef, scaled_leaves)

        return scaled_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def preconditioned_sgld(
    key: jax.Array,
    learning_rate: optax.ScalarOrSchedule,
    temperature: float = 1.0,
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    warmup_steps=100,
) -> optax.GradientTransformation:
    """
    Preconditioned SGLD with RMSprop.

    The update rule is:
        θ_{t+1} = θ_t + (ϵ_t/2)[G(θ_t)(∇log p(θ_t) + N̄g(θ_t; D^t)) + Γ(θ_t)] + N(0, ϵ_t G(θ_t))

    where:
        - G(θ_t) is the preconditioning matrix (from RMSprop): diag(1 ⊙ (λI + √V(θ_t)))
        - V(θ_t) is the running average of squared gradients
        - N(0, ϵ_t G(θ_t)) is Gaussian noise with covariance ϵ_t G(θ_t)

    Args:
        learning_rate: Learning rate ϵ_t
        temperature: Temperature parameter (default 1.0)
        decay: RMSprop decay rate α (typically 0.9 or 0.99)
        eps: Small constant for numerical stability (λ in the algorithm)
        initial_scale: Initial value for second moment.
        warmup_steps: num steps without noise (to warmup moving average)
        key: Random key for noise generation

    Returns:
        A GradientTransformation implementing preconditioned SGLD with RMSprop.
    """

    if callable(learning_rate):
        lr_schedule = learning_rate
    else:
        lr_schedule = lambda step: learning_rate

    # RMSprop uses scale_by_rms which computes: update / sqrt(v + eps)
    # where v is the running mean of squared gradients
    # This is equivalent to multiplying by G(θ) = diag(1 / sqrt(v + eps))

    scale_by_rms_transform = optax.scale_by_rms(
        decay=decay, eps=eps, initial_scale=initial_scale
    )

    def init_fn(params):
        # Initialize RMSprop state (running mean of squared gradients)
        return {
            "scale_by_rms_state": scale_by_rms_transform.init(params),
            "noise_state": optax.AddNoiseState(
                count=jnp.zeros([], jnp.int32), rng_key=key
            ),
        }

    def update_fn(updates, state, params=None):
        # Warmup: no noise for first N steps
        noise_fraction = jax.lax.cond(
            state["noise_state"].count < warmup_steps,
            lambda: jnp.clip(state["noise_state"].count / warmup_steps, 0.0, 1.0),
            lambda: jnp.array(1.0, dtype=jnp.float32),
        )

        # Step 1: Apply RMSprop preconditioning to gradients
        # This computes: preconditioned_grad = grad / sqrt(v + eps) = G(θ) * grad
        preconditioned_updates, new_rms_state = scale_by_rms_transform.update(
            updates, state["scale_by_rms_state"], params
        )

        # Step 2: Generate noise with covariance proportional to G(θ)
        # Noise should be: N(0, ϵ_t * G(θ_t))
        # Since G(θ) = diag(1 / sqrt(v + eps)), the noise std for each parameter is:
        # sqrt(ϵ_t / sqrt(v + eps)) = sqrt(ϵ_t) / (v + eps)^(1/4)

        count_inc = state["noise_state"].count + 1
        current_lr = lr_schedule(state["noise_state"].count)

        # Get the RMSprop statistics (v = running mean of squared gradients)
        # The scale_by_rms state contains 'nu' which is the second moment estimate
        v = state["scale_by_rms_state"].nu

        # Generate standard normal noise matching the structure of updates
        rng_key, sample_key = jax.random.split(state["noise_state"].rng_key)
        noise = optax.tree.random_like(
            sample_key, target_tree=updates, sampler=jax.random.normal
        )

        # Scale noise by sqrt(ϵ_t * temperature) / (v + eps)^(1/4)
        # This gives noise ~ N(0, ϵ_t * temperature * G(θ))
        noise_scale = jnp.sqrt(current_lr * temperature)
        min_scale = 0.1  # Prevents noise explosion
        preconditioned_noise = jax.tree_util.tree_map(
            lambda n, v_param: noise_scale
            * n
            / jnp.maximum(jnp.power(v_param + eps, 0.25), min_scale),
            noise,
            v,
        )

        # Step 3: Combine preconditioned gradients and noise
        # Final update: (ϵ_t/2) * G(θ) * grad + sqrt(ϵ_t) * G(θ)^(1/2) * noise
        # Note: we apply learning_rate / 2 scaling
        final_updates = jax.tree_util.tree_map(
            lambda pg, pn: -(current_lr / 2.0) * pg + noise_fraction * pn,
            preconditioned_updates,
            preconditioned_noise,
        )

        new_noise_state = optax.AddNoiseState(count=count_inc, rng_key=rng_key)
        new_state = {
            "scale_by_rms_state": new_rms_state,
            "noise_state": new_noise_state,
        }

        return final_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def sgld(
    key: jax.Array,
    learning_rate: optax.ScalarOrSchedule,
    temperature: float = 1.0,
    adaptive_noise: bool = True,
) -> optax.GradientTransformation:
    """
    Standard Stochastic Gradient Langevin Dynamics (SGLD).

    The update rule is:
        θ_{t+1} = θ_t + ϵ_t * ∇log p(θ_t) + N(0, 2 * ϵ_t * temperature * I)

    This implements the standard SGLD algorithm which samples from the posterior
    p(θ|D) ∝ exp(-U(θ)/temperature) where U(θ) is the potential (negative log-posterior).

    The noise term ensures detailed balance for proper MCMC sampling. The learning rate
    must satisfy Robbins-Monro conditions: Σϵ_t = ∞ and Σϵ_t² < ∞.

    Args:
        learning_rate: Learning rate schedule ϵ_t (must decay over time)
        temperature: Base temperature parameter (default 1.0 for posterior sampling).
            When adaptive_noise=True, this is the base temperature used to compute
            adaptive temperature based on gradient norm.
        adaptive_noise: If True, scale noise by adaptive temperature based on gradient norm.
            Adaptive temperature = base_temp / (1.0 + grad_norm). This reduces noise when
            gradients are large (more stable) and increases noise when gradients are small
            (more exploration). Default True.
        key: Random key for noise generation

    Returns:
        A GradientTransformation implementing standard SGLD.

    Example:
        >>> key = jr.key(0)
        >>> lr_schedule = lambda t: 0.01 * (1.0 + t / 100.0) ** -0.5
        >>> optimizer = sgld(key, lr_schedule, temperature=1.0, adaptive_noise=True)
    """

    if callable(learning_rate):
        lr_schedule = learning_rate
    else:
        lr_schedule = lambda step: learning_rate

    def init_fn(params):
        # Only need to track step count and RNG key (no preconditioner state)
        return optax.AddNoiseState(count=jnp.zeros([], jnp.int32), rng_key=key)

    def update_fn(updates, state, params=None):
        # Get current learning rate from schedule
        count_inc = state.count + 1
        current_lr = lr_schedule(state.count)

        # Clip learning rate to prevent numerical issues
        current_lr = jnp.clip(current_lr, 1e-6, 1.0)

        # Compute adaptive temperature based on gradient norm if enabled
        if adaptive_noise:
            # Compute gradient norm (L2 norm of all gradient elements)
            flat_grads, _ = jax.flatten_util.ravel_pytree(updates)
            grad_norm = jnp.linalg.norm(flat_grads)

            # Adaptive temperature: reduces noise when gradients are large
            # When grad_norm is large → adaptive_temp is small → less noise (more stable)
            # When grad_norm is small → adaptive_temp ≈ base_temp → more noise (more exploration)
            adaptive_temp = temperature / (1.0 + grad_norm)
        else:
            adaptive_temp = temperature

        # Generate standard normal noise matching the structure of updates
        rng_key, sample_key = jax.random.split(state.rng_key)
        noise = optax.tree.random_like(
            sample_key, target_tree=updates, sampler=jax.random.normal
        )

        # Standard SGLD update:
        # θ_{t+1} = θ_t + ϵ_t * grad + N(0, 2 * ϵ_t * adaptive_temp * I)
        # The noise std is sqrt(2 * ϵ_t * adaptive_temp)
        noise_std = jnp.sqrt(2.0 * current_lr * adaptive_temp)

        # Scale noise and combine with gradient update
        scaled_noise = jax.tree_util.tree_map(lambda n: noise_std * n, noise)

        # Final update: gradient step + noise
        # Clip updates to prevent extreme values that could break the simulation
        final_updates = jax.tree_util.tree_map(
            lambda g, n: jnp.clip(-current_lr * g + n, -10.0, 10.0),
            updates,
            scaled_noise,
        )

        new_state = optax.AddNoiseState(count=count_inc, rng_key=rng_key)
        return final_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def hybrid_optimizer(
    ode_optim=optax.adam,
    nn_optim=optax.adamw,
    ode_vs_nn_labels=None,
    model_param_transform=None,
):
    """
    Hybrid optimizer for HH + Neural ODE models.

    Args:
        ode_optim: Optimizer for HH/ODE parameters
        nn_optim: Optimizer for neural network parameters
        model_param_transform: Optional ModelParamTransform object.
    """
    ode_transforms = [
        optax.zero_nans(),
        # optax.scale(1.0),
        # scale_by_transform_magnitude(epsilon=1e-8, model_param_transform=model_param_transform),
        # scale_by_param_magnitude(epsilon=1e-8),
        # optax.scale_by_param_block_norm(min_scale=1e-3),
        optax.clip_by_global_norm(0.1),
        ode_optim,
    ]

    neural_transforms = [
        optax.zero_nans(),
        # optax.scale(1.0),
        # optax.scale_by_param_block_norm(min_scale=1e-3),
        # optax.clip_by_global_norm(0.5),
        optax.clip_by_global_norm(0.1),
        nn_optim,
    ]

    joint_optimizer = optax.multi_transform(
        {"ode": optax.chain(*ode_transforms), "nn": optax.chain(*neural_transforms)},
        param_labels=ode_vs_nn_labels,
    )
    optimizer = optax.chain(
        # optax.scale_by_param_block_norm(min_scale=1e-3),
        joint_optimizer,
    )

    return optimizer


def mask_gradients(grads, model_part="nn"):
    """
    Mask gradients for staged training of hybrid models.

    Args:
        grads: Gradient tree structure
        model_part: str, default="ode"
            - "ode": Mask out ODE gradients, train NN only
            - "nn": Mask out NN gradients, train ODE only

    """

    def apply_mask(path, grad):
        is_nn = any("net" in str(p) for p in path)
        if model_part == "nn":
            return grad * (1.0 - is_nn)
        elif model_part == "ode":
            return grad * is_nn
        else:
            raise ValueError(f"model_part must be 'ode' or 'nn', got {model_part}")

    return jax.tree_util.tree_map_with_path(apply_mask, grads, is_leaf=eqx.is_array)


def zero_masked_weights(model):
    """
    Zero out masked weights in MaskedLinear layers after updates.

    This prevents weight decay in AdamW from causing masked weights to drift
    away from zero. Should be called after each optimizer update.

    Note: Weight decay in AdamW applies to all weights, including masked ones.
    While masked weights don't affect the output, weight decay can cause them
    to drift. This function ensures they stay at zero.

    Args:
        model: Model containing MaskedLinear layers

    Returns:
        Model with masked weights set to zero
    """
    from hybrid_models.hh.channels import MaskedLinear

    def is_masked_linear(x):
        return isinstance(x, MaskedLinear)

    def update_masked_linear(ml):
        # Zero weights where mask is 0
        # MaskedLinear always has a mask, so we can access it directly
        masked_weight = ml.weight * ml.mask
        return eqx.tree_at(lambda x: x.weight, ml, masked_weight)

    # Partition model into MaskedLinear and everything else
    masked_linears, rest = eqx.partition(
        model, is_masked_linear, is_leaf=is_masked_linear
    )

    # Update only the MaskedLinear layers
    updated_masked_linears = jax.tree.map(
        update_masked_linear, masked_linears, is_leaf=is_masked_linear
    )

    # Combine back
    return eqx.combine(updated_masked_linears, rest)


def jacobian_penalty_hutchinson(model, t, u, key, state_keys=None):
    """Approximate ||J||_F^2 via E[||J v||^2], using one Hutchinson probe.

    model: callable (t, u, args) -> dict of derivatives (same keys as u plus extra like i_* possibly)
    t: scalar time
    u: dict state (e.g. {"v": ..., "x1": ..., ...})
    key: PRNGKey
    """
    # only differentiate wrt chosen state variables
    u_state = {k: u[k] for k in state_keys}
    u_flat, unravel = jax.flatten_util.ravel_pytree(u_state)

    v = jr.rademacher(key, u_flat.shape).astype(u_flat.dtype)

    def f_flat(u_flat_):
        u_state_ = unravel(u_flat_)
        u_full = dict(u)
        u_full.update(u_state_)

        du = model(t, u_full, None)

        # only penalize corresponding outputs
        du_state = {k: du[k] for k in state_keys}
        du_flat, _ = jax.flatten_util.ravel_pytree(du_state)
        return du_flat

    Jv = jax.jvp(f_flat, (u_flat,), (v,))[1]
    return jnp.mean(Jv**2)