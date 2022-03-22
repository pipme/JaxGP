from abc import abstractmethod
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax import vmap

from .config import Config
from .helpers import Array, dataclass, field


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


@dataclass
class Kernel:
    active_dims: Tuple[int] = (0,)
    stationary: Optional[bool] = False
    spectral: Optional[bool] = False
    name: Optional[str] = "Kernel"
    _params: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ndims = len(self.active_dims)

    @abstractmethod
    def __call__(self, x: Array, y: Array, params: dict) -> Array:
        raise NotImplementedError

    def slice_input(self, x: Array) -> Array:
        return x[..., self.active_dims]

    @property
    def ard(self) -> bool:
        return True if self.ndims > 1 else False

    @property
    def params(self) -> dict:
        return self._params

    @property
    @abstractmethod
    def transforms(self) -> Dict:
        raise NotImplementedError


@dataclass
class RBF(Kernel):
    name: Optional[str] = "Radial basis function kernel"
    distance: Distance = L2Distance()

    def __post_init__(self) -> None:
        self.ndims = len(self.active_dims)
        self._params = {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "outputscale": jnp.array([1.0]),
        }

    def __call__(self, x: Array, y: Array, params: dict) -> Array:
        for key, _ in self._params.items():
            assert self._params[key].shape == params[key].shape
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["outputscale"] * jnp.exp(
            -0.5 * self.distance.squared_distance(x, y)
        )
        return K.squeeze()

    @property
    def transforms(self) -> Dict:
        return {
            "outputscale": Config.positive_bijector,
            "lengthscale": Config.positive_bijector,
        }


def gram(
    kernel: Kernel, inputs: Array, params: dict, full_cov: bool = True
) -> Array:
    """Compute gram matrix of the inputs."""
    if full_cov:
        return vmap(
            lambda x1: vmap(lambda y1: kernel(x1, y1, params))(inputs)
        )(inputs)
    else:
        return vmap(lambda x: kernel(x, x, params))(inputs)


def cross_covariance(
    kernel: Kernel, X: Array, Y: Array, params: dict
) -> Array:
    """Compute covariance matrix between X and Y."""
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(Y))(X)
