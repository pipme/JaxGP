from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import vmap

from .config import Config
from .helpers import Array, dataclass, field

# TODO: more friendly api for kernel class, split to stationary kernels etc.


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


class Kernel(metaclass=ABCMeta):
    if TYPE_CHECKING:

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    @abstractmethod
    def __call__(self, x: Array, y: Array, params: Dict) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self) -> Dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def transforms(self) -> Dict:
        raise NotImplementedError


@dataclass
class Stationary(Kernel):
    active_dims: Tuple[int] = (0,)
    stationary: bool = False
    spectral: bool = False
    name: str = "Stationary Kernel"

    def __post_init__(self) -> None:
        if len(jnp.unique(jnp.array(self.active_dims))) != len(
            self.active_dims
        ):
            raise ValueError("active_dims should contain unique indices.")
        assert jnp.all(jnp.array(self.active_dims) >= 0)

    def slice_input(self, x: Array) -> Array:
        return x[..., self.active_dims]

    @property
    def ard(self) -> bool:
        return True if self.ndims > 1 else False

    @property
    def ndims(self) -> int:
        return len(self.active_dims)


@dataclass
class RBF(Stationary):
    name: str = "Radial basis function kernel (ARD)"
    distance: Distance = L2Distance()

    def __call__(self, x: Array, y: Array, params: Dict) -> Array:
        assert max(self.active_dims) < x.shape[-1]
        for key, _ in self.params.items():
            assert self.params[key].shape == params[key].shape
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["outputscale"] * jnp.exp(
            -0.5 * self.distance.squared_distance(x, y)
        )
        return K.squeeze()

    @property
    def params(self) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "outputscale": jnp.array([1.0]),
        }

    @property
    def transforms(self) -> Dict:
        return {
            "outputscale": Config.positive_bijector,
            # "lengthscale": Config.positive_bijector,
            "lengthscale": Config.strict_positive_bijector,
        }


def gram(
    kernel: Stationary, X: Array, params: Dict, full_cov: bool = True
) -> Array:
    """Compute gram matrix of the inputs."""
    if isinstance(kernel, RBF):
        if full_cov:
            X = X / params["lengthscale"]
            dist = euclidean_distance(X)
            # i, j = jnp.diag_indices(min(dist.shape[-2:]))
            # dist = dist.at[..., i, j].set(0.)
            return params["outputscale"] * jnp.exp(-0.5 * dist)
        else:
            return params["outputscale"] * jnp.ones(X.shape[0])
    if full_cov:
        return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(X))(X)
    else:
        return vmap(lambda x: kernel(x, x, params))(X)


def cross_covariance(
    kernel: Stationary, X: Array, Y: Array, params: Dict
) -> Array:
    """Compute covariance matrix between X and Y."""

    if isinstance(kernel, RBF):
        # more efficient than vmap, vmap is similar to broadcasting
        # "jnp.sum((X[:, None, :] - Y[None, :, :])**2, -1)"
        X = X / params["lengthscale"]
        Y = Y / params["lengthscale"]
        dist = euclidean_distance(X, Y)
        return params["outputscale"] * jnp.exp(-0.5 * dist)
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(Y))(X)


def euclidean_distance(X, Y=None):
    if Y is None:
        mu = X.mean(0)
        X -= mu
        Xs = jnp.sum(X**2, -1)
        dist = -2 * X @ X.T + Xs[:, None] + Xs[None, :]
    else:
        n = X.shape[0]
        m = Y.shape[0]
        mu = n / (m + n) * X.mean(0) + m / (m + n) * Y.mean(0)
        X -= mu
        Y -= mu
        Xs = jnp.sum(X**2, -1)
        X2s = jnp.sum(Y**2, -1)
        dist = -2 * X @ Y.T + Xs[:, None] + X2s[None, :]
    dist = jnp.maximum(dist, 0.)
    return dist