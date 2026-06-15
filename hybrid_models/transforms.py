from abc import ABC, abstractmethod
from typing import List, Dict

import jax
import jax.numpy as jnp

from jax import Array
from jax.typing import ArrayLike

from hybrid_models.utils import tree_apply_with_path


# the code below was taken from the jaxley simulator.
class Transform(ABC):
    def __call__(self, x: ArrayLike) -> Array:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ArrayLike) -> Array:
        pass

    @abstractmethod
    def inverse(self, x: ArrayLike) -> Array:
        pass


class SigmoidTransform(Transform):
    """Sigmoid transformation."""

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        """This transform maps any value bijectively to the interval [lower, upper].

        Args:
            lower (ArrayLike): Lower bound of the interval.
            upper (ArrayLike): Upper bound of the interval.
        """
        super().__init__()
        self.lower = lower
        self.width = upper - lower

    def forward(self, x: ArrayLike) -> Array:
        y = 1.0 / (1.0 + jnp.exp(-x))
        return self.lower + self.width * y

    def inverse(self, y: ArrayLike) -> Array:
        x = (y - self.lower) / self.width
        x = -jnp.log((1.0 / x) - 1.0)
        return x


class LogisticTransform(SigmoidTransform):
    """Logistic transformation."""

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        """This transform maps any value bijectively to the interval [lower, upper].

        Args:
            lower (ArrayLike): Lower bound of the interval.
            upper (ArrayLike): Upper bound of the interval.
        """
        super().__init__(lower, upper)

    def forward(self, x: ArrayLike) -> Array:
        x_logit = x * jnp.pi / jnp.sqrt(3)  # rescale (logistic has variance π²/3)
        return super().forward(x_logit)

    def inverse(self, y: ArrayLike) -> Array:
        x_logit = super().inverse(y)
        return x_logit * jnp.sqrt(3) / jnp.pi  # Scale to unit variance


class SoftplusTransform(Transform):
    """Softplus transformation."""

    def __init__(self, lower: ArrayLike) -> None:
        """This transform maps any value bijectively to the interval [lower, inf).

        Args:
            lower (ArrayLike): Lower bound of the interval.
        """
        super().__init__()
        self.lower = lower

    def forward(self, x: ArrayLike) -> Array:
        return jnp.log1p(jnp.exp(x)) + self.lower

    def inverse(self, y: ArrayLike) -> Array:
        return jnp.log(jnp.exp(y - self.lower) - 1.0)


class IdentityTransform(Transform):
    """Identity transformation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: ArrayLike) -> Array:
        return x

    def inverse(self, y: ArrayLike) -> Array:
        return y


class ParamTransform:
    """Parameter transformation utility.

    This class is used to transform parameters usually from an unconstrained space to a constrained space
    and back (bacause most biophysical parameter are bounded). The user can specify a PyTree of transforms
    that are applied to the parameters.

    Attributes:
        tf_dict: A PyTree of transforms for each parameter.

    """

    def __init__(self, tf_dict: List[Dict[str, Transform]] | Transform) -> None:
        """Creates a new ParamTransform object.

        Args:
            tf_dict: A PyTree of transforms for each parameter.
        """

        self.tf_dict = tf_dict

    def forward(
        self, params: List[Dict[str, ArrayLike]] | ArrayLike
    ) -> Dict[str, Array]:
        """Pushes unconstrained parameters through a tf such that they fit the interval.

        Args:
            params: A list of dictionaries (or any PyTree) with unconstrained parameters.

        Returns:
            A list of dictionaries (or any PyTree) with transformed parameters.

        """

        return jax.tree_util.tree_map(lambda x, tf: tf.forward(x), params, self.tf_dict)

    def inverse(
        self, params: List[Dict[str, ArrayLike]] | ArrayLike
    ) -> Dict[str, Array]:
        """Takes parameters from within the interval and makes them unconstrained.

        Args:
            params: A list of dictionaries (or any PyTree) with transformed parameters.

        Returns:
            A list of dictionaries (or any PyTree) with unconstrained parameters.
        """

        return jax.tree_util.tree_map(lambda x, tf: tf.inverse(x), params, self.tf_dict)


class ModelParamTransform:
    def __init__(self, param_tf):
        self.param_tf = param_tf

    def forward(self, module):
        forward_tfs = {k: v.forward for k, v in self.param_tf.tf_dict.items()}
        return tree_apply_with_path(module, forward_tfs)

    def inverse(self, module):
        inverse_tfs = {k: v.inverse for k, v in self.param_tf.tf_dict.items()}
        return tree_apply_with_path(module, inverse_tfs)


class BoxConstraint:
    def __init__(self, bounds):
        self._bounds = bounds

    def _get_all_param_names(self, module):
        """Extract all parameter names from the module"""
        leaves, _ = jax.tree_util.tree_flatten_with_path(module)
        param_names = set()
        for path, leaf in leaves:
            if path and hasattr(path[-1], "name"):
                param_names.add(path[-1].name)
        return param_names

    def bounds(self, module):
        param_names = self._get_all_param_names(module)
        upper_bounds = {}
        lower_bounds = {}

        for name in param_names:
            if name in self._bounds:
                lower_bound = self._bounds[name][0]
                upper_bound = self._bounds[name][1]
                lower_bounds[name] = lambda x, bound=lower_bound: bound
                upper_bounds[name] = lambda x, bound=upper_bound: bound
            else:
                lower_bounds[name] = lambda x: None
                upper_bounds[name] = lambda x: None

        return tree_apply_with_path(module, lower_bounds), tree_apply_with_path(
            module, upper_bounds
        )
