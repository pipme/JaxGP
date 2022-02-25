from abc import abstractmethod
from email.policy import default
from chex import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from pytest import param
from .types import Array
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial


class Distance:
    """An abstract base class defining a distance metric interface"""

    def distance(self, X1: Array, X2: Array) -> Array:
        """Compute the distance between two coordinates under this metric"""
        raise NotImplementedError()

    def squared_distance(self, X1: Array, X2: Array) -> Array:
        """Compute the squared distance between two coordinates

        By default this returns the squared result of
        :func:`tinygp.kernels.stationary.Distance.distance`, but some metrics
        can take advantage of these separate implementations to avoid
        unnecessary square roots.
        """
        return jnp.square(self.distance(X1, X2))


class L1Distance(Distance):
    """The L1 or Manhattan distance between two coordinates"""

    def distance(self, X1: Array, X2: Array) -> Array:
        return jnp.sum(jnp.abs(X1 - X2))


class L2Distance(Distance):
    """The L2 or Euclidean distance bettwen two coordaintes"""

    def distance(self, X1: Array, X2: Array) -> Array:
        return jnp.sqrt(self.squared_distance(X1, X2))

    def squared_distance(self, X1: Array, X2: Array) -> Array:
        return jnp.sum(jnp.square(X1 - X2))


@dataclass(repr=False)
class Kernel:
    active_dims: Optional[List[int]] = None
    stationary: Optional[bool] = False
    spectral: Optional[bool] = False
    name: Optional[str] = "Kernel"
    _params: Optional[Dict] = None

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    @abstractmethod
    def __call__(self, x: Array, y: Array, params: dict) -> Array:
        raise NotImplementedError

    def slice_input(self, x: Array) -> Array:
        return x[..., self.active_dims]

    @property
    def ard(self):
        return True if self.ndims > 1 else False

    @property
    def params(self) -> dict:
        return self._params

    @params.setter
    def params(self, value):
        self._params = value


@dataclass(repr=False)
class RBF(Kernel):
    name: Optional[str] = "Radial basis function kernel"
    distance: Distance = L2Distance()

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)
        self._params = {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }

    def __call__(
        self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict
    ) -> Array:
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * jnp.exp(-0.5 * self.distance(x, y))
        return K.squeeze()


# @partial(jit, static_argnames="diag")
def gram(
    kernel: Kernel, inputs: Array, params: dict, diag: bool = False
) -> Array:
    if diag:
        return vmap(lambda x: kernel(x, x, params))(inputs)
    else:
        return vmap(
            lambda x1: vmap(lambda y1: kernel(x1, y1, params))(inputs)
        )(inputs)


def cross_covariance(
    kernel: Kernel, x: Array, y: Array, params: dict
) -> Array:
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(x))(y)
