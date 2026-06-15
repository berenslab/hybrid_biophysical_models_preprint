import jax.numpy as jnp
import numpy as np

from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController
import jax
import equinox as eqx
import json
import inspect
import logging

rng = lambda: jnp.array(np.random.randn())


def safe_log(
    x, min_val=1e-10, neg_inf_if_below=True
):  # ~6e-5 fp16, ~1e-10 fp32, ~1e-20 fp64
    return jnp.where(
        x > min_val, jnp.log(x), jnp.log(min_val) if neg_inf_if_below else -jnp.inf
    )


def safe_exp(x, max_val=85.0):  # ~11 fp16, ~85 fp32, ~700 fp64
    # exp = jnp.where(x < max_val, jnp.exp(x), jnp.exp(max_val)) # prev. behaviour
    exp = jnp.exp(jnp.clip(x, a_min=None, a_max=max_val))
    return exp


def safe_expm1(x, max_val=85.0):  # ~11 fp16, ~85 fp32, ~700 fp64
#     expm1 =  jnp.where(x < max_val, jnp.expm1(x), jnp.expm1(max_val)) # prev. behaviour
    expm1 = jnp.expm1(jnp.clip(x, a_min=None, a_max=max_val))
    return expm1


def efun(z):
#     efn = jnp.where(jnp.abs(z) < 1e-4, 1 - z / 2, z / safe_expm1(z)) # prev. behaviour
    safe_z = jnp.where(jnp.abs(z) < 1e-4, jnp.ones_like(z), z)  # avoid 0/0 in inactive branch
    efn = jnp.where(jnp.abs(z) < 1e-4, 1 - z / 2, safe_z / safe_expm1(safe_z))
    return efn


def vtrap(x, y):
    xy = x / y
#     vtrap = x / (safe_exp(x / y) - 1.0) # prev. behaviour
    safe_xy = jnp.where(jnp.abs(xy) < 1e-6, jnp.ones_like(xy), xy)  # avoid 0/0 in inactive branch
    vtrap = jnp.where(jnp.abs(xy) < 1e-6, y * (1.0 - xy / 2.0), x / safe_expm1(safe_xy))
    return vtrap


class Uniform:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def sample(self, key, shape=(1,)):
        return jax.random.uniform(
            key, shape=shape, minval=self.lower, maxval=self.upper
        )

    def log_prob(self, x):
        in_bounds = (x >= self.lower) & (x <= self.upper)
        return jnp.where(in_bounds, -jnp.log(self.upper - self.lower), -jnp.inf)


class ProductDistribution:
    def __init__(self, dists):
        self.dists = dists

    def sample(self, key, shape=(1,)):
        split_keys = jax.random.split(key, len(self.dists))
        split_keys = {k: v for k, v in zip(self.dists.keys(), split_keys)}
        return jax.tree_util.tree_map(
            lambda k, d: d.sample(k, shape), split_keys, self.dists
        )

    def log_prob(self, x):
        log_probs = jax.tree_util.tree_map(lambda x, d: d.log_prob(x), x, self.dists)
        return jnp.sum(jnp.array(jax.tree_leaves(log_probs)), axis=0)


class BoxUniform(ProductDistribution):
    def __init__(self, bounds):
        dists = {k: Uniform(*v) for k, v in bounds.items()}
        super().__init__(dists)


def tree_path_of_leaves(model, filter: str | list[str] = None, invert=False):
    # TODO: allow partial matches, like .channels -> should show all channels
    # TODO: prevent .c from matching .channels as well
    def label_fn(path, value):
        path_str = "".join([str(p) for p in path])
        cond = (path_str in filter) if filter else True
        return path_str if cond != invert else None

    return jax.tree_util.tree_map_with_path(label_fn, model)


def tree_filter_by_path(model, filter: str | list[str], invert=False):
    # TODO: add option for is_leaf, i.e. to allow filtering partial paths, like .channels -> should show all channels
    def label_fn(path, value):
        path_str = "".join([str(p) for p in path])
        return value if (path_str in filter) != invert else None

    return jax.tree_util.tree_map_with_path(label_fn, model)


def tree_apply_with_path(model, apply_dict):
    def label_fn(path, value):
        path_str = "".join([str(p) for p in path])
        return apply_dict.get(path_str, lambda x: None)(value)

    new_vals = jax.tree.map(lambda x: None, model)
    new_vals = eqx.combine(new_vals, jax.tree_util.tree_map_with_path(label_fn, model))

    old_vals = jax.tree.map(
        lambda old, new: old if new is None else None, model, new_vals
    )
    return eqx.combine(old_vals, new_vals)


def tree_set_with_path(model, set_dict):
    return tree_apply_with_path(
        model, {k: (lambda v: lambda _: v)(v) for k, v in set_dict.items()}
    )


class ProgressBar:
    def __init__(self, fmt=None, pbar_segments=10):
        self.filled_segments = 0
        self.total_segments = pbar_segments
        self.step = 0
        self.fmt = fmt if fmt is not None else "{self}"

    @property
    def empty_segments(self):
        return self.total_segments - self.filled_segments

    def __str__(self):
        return f"|{'█' * self.filled_segments}{' ' * self.empty_segments}|"

    def render(self, **fmt_kwargs):
        return self.fmt.format(self=self, **fmt_kwargs)

    def __call__(self, percent, **fmt_kwargs):
        self.filled_segments = int(self.total_segments * percent)
        return self.render(**fmt_kwargs)


def label_params(model, map_if_path_contains={"net": "nn"}, otherwise="ode"):
    """Label parameters as 'ode' or 'nn' depending on whether they are nn or ode params."""

    def label_fn(path, value):
        for key, label in map_if_path_contains.items():
            if any(key in str(p) for p in path):
                return label
        return otherwise

    labels = jax.tree_util.tree_map_with_path(label_fn, model)
    return labels


def filter_params_by_label(params, labels, target_label):
    """Filter a PyTree to only keep leaves with a specific label."""
    return jax.tree_util.tree_map(
        lambda param, label: param if label == target_label else None, params, labels
    )


class Dataset:
    def __init__(self, X, Y, metadata=None):
        self.X = X
        self.Y = Y
        self.metadata = metadata

    def __len__(self):
        return len(jax.tree.leaves(self.X)[0])

    def __getitem__(self, idx):
        x_i = jax.tree.map(lambda x: jnp.atleast_1d(x)[idx], self.X)
        y_i = jax.tree.map(lambda y: jnp.atleast_1d(y)[idx], self.Y)
        return x_i, y_i

    def to_file(self, filename):
        data = {}
        data["X"] = self.X
        data["Y"] = self.Y
        data["metadata"] = self.metadata
        with open(filename, "w") as f:
            json.dump(data, f, cls=CustomJSONEncoder, indent=2)

    @staticmethod
    def from_file(filename):
        def try_array(x):
            if isinstance(x, list):
                if not isinstance(x[0], str):
                    return jnp.array(x)
            return x

        with open(filename, "r") as f:
            data = json.load(f)
            data = jax.tree.map(try_array, data, is_leaf=lambda x: isinstance(x, list))
        return Dataset(**data)


class DataLoader:
    def __init__(
        self,
        dataset,
        shuffle=True,
        batch_size=1,
        key=jax.random.key(
            0
        ),  # Note: utils.py doesn't import jax.random as jr, so using jax.random.key
        cycle_batches=False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size if batch_size > 0 else len(dataset)
        self.key = key
        self.subkey = key
        self.cycle_batches = cycle_batches

    def __iter__(self):
        order = jnp.arange(len(self.dataset))
        self.subkey, subkey = jax.random.split(self.subkey)
        if self.shuffle:
            order = jax.random.permutation(subkey, order)

        if self.cycle_batches:
            while True:
                for i in range(0, len(self.dataset), self.batch_size):
                    yield self.dataset[order[i : i + self.batch_size]]
                if self.shuffle:
                    subkey, subsubkey = jax.random.split(subkey)
                    order = jax.random.permutation(subsubkey, order)

        else:
            for i in range(0, len(self.dataset), self.batch_size):
                yield self.dataset[order[i : i + self.batch_size]]


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if callable(obj) or inspect.isclass(obj):
                return str(obj)
            elif isinstance(obj, jnp.ndarray):
                return obj.tolist()
            elif isinstance(obj, eqx.Module):
                return {
                    "type": obj.__class__.__name__,
                    "fields": {k: self.default(getattr(obj, k)) for k in obj.__dataclass_fields__.keys()},
                }
        except:
            return f"<non-serializable: {type(obj).__name__}>"


def block_stack(blocks, pad_value=0.0, dtype=None):
    if dtype is None:
        dtype = blocks[0].dtype
    shape = tuple(sum(sizes) for sizes in zip(*[x.shape for x in blocks]))
    X = jnp.full(shape, pad_value, dtype=dtype)
    block_i0, block_j0 = 0, 0
    for i, block in enumerate(blocks):
        b_height, b_width = block.shape
        X = X.at[block_i0 : block_i0 + b_height, block_j0 : block_j0 + b_width].set(
            block
        )
        block_i0 += b_height
        block_j0 += b_width
    return X


def setup_logging(fout, to_fout=True, to_stdout=False):
    """
    Set up logging configuration with file and/or console handlers.

    Args:
        fout: Full output file path including .log extension for log file
        to_fout: If True, log to file (default: True)
        to_stdout: If True, also add console handler that filters progress bar messages (default: False)
    """
    logger = logging.getLogger()
    # Remove all existing handlers to prevent accumulation
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Configure formatter
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    class PBarFilter(logging.Filter):
        def filter(self, record):
            return not record.getMessage().startswith("Epoch ")

    # To avoid duplicate output to terminal when to_fout is False and to_stdout is True,
    # only add a stream handler (stdout) once, and add the pbar-filtering handler only if both are requested.
    if to_fout:
        file_handler = logging.FileHandler(fout, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Optionally, also log to stdout if requested
        if to_stdout:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.addFilter(PBarFilter())
            console_handler.setFormatter(file_formatter)
            logger.addHandler(console_handler)
    else:
        # No file output, just log to stdout (do not double-add handler if to_stdout is also True)
        stdout_handler = logging.StreamHandler()
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(file_formatter)
        logger.addHandler(stdout_handler)
        logger.setLevel(logging.INFO)
        # If to_stdout True, add PBarFilter to the only handler so that PBar messages are filtered out
        if to_stdout:
            stdout_handler.addFilter(PBarFilter())


def fmt_elapsed_time(elapsed_time):
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    return f"{hours:02d}h:{minutes:02d}m:{seconds:05.2f}s"


def assert_finite(model, params=None):
    """
    Check if all parameters are finite.
    """

    def _is_finite(x):
        return jnp.isfinite(x) if isinstance(x, jax.Array) else True

    if params is not None:
        model = model.set(params)  # this should only set static params!

    non_finite_paths = jax.tree.map(_is_finite, model)
    paths = jax.tree_util.tree_leaves_with_path(non_finite_paths)
    non_finite_paths = {}
    for path, v in paths:
        if not jnp.all(v):
            non_finite_paths["".join([str(p) for p in path])] = v

    if non_finite_paths:
        raise ValueError(f"Parameters are not finite at paths: {non_finite_paths}")


def absorb_kwargs(*names):
    """
    Wrap a function to remove specific keyword arguments (`names`) before calling the original function.

    Args:
        *names: Names of keyword arguments to remove from kwargs before passing to fn.
    Returns:
        A decorator that cleans kwargs and calls fn.
    """

    def decorator(fn):
        def wrapped(*args, **kwargs):
            stripped_kwargs = {k: v for k, v in kwargs.items() if k not in names}
            return fn(*args, **stripped_kwargs)

        return wrapped

    return decorator

def normal_initializer(m=0.0, std=0.01):
    def initializer(key, shape):
        return m + std * jax.random.normal(key, shape)

    return initializer

def custom_initializer(weights_diag, bias=None, off_diag_init_fn=None):
    def initializer(key, shape):
        if len(shape) == 2:
            if off_diag_init_fn is not None:
                weights = off_diag_init_fn(key, shape)
            else:
                weights = jnp.zeros(shape)
            return weights.at[jnp.diag_indices(min(shape))].set(weights_diag)
        elif bias is not None and len(shape) == 1:
            return jnp.broadcast_to(jnp.array(bias), shape)
        else:
            raise ValueError(f"Shape {shape} does not match provided weights or bias.")

    return initializer



def flatten_dict(d, parent_key='', sep='//'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def nest_flattened_dict(flat_dict, sep='//'):
    """Unflattens a dictionary, creating a nested dictionary based on the separator."""
    d = {}
    for k, v in flat_dict.items():
        keys = k.split(sep)
        current = d
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = v
    return d

def update_config(target_obj, updates):
    for k, v in updates.items():
        if hasattr(target_obj, k):
            current = getattr(target_obj, k)
            if isinstance(v, dict) and isinstance(current, dict):
                current.update(v)
            elif isinstance(v, dict) and hasattr(current, "__dict__"):
                update_config(current, v)
            else:
                setattr(target_obj, k, v)
        else:
            setattr(target_obj, k, v)