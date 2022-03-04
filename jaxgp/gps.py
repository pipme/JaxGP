from abc import abstractmethod, abstractproperty
from typing import Callable, Dict, Optional

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from chex import dataclass
from jax.scipy import linalg

from .kernels import Kernel, cross_covariance, gram
from .likelihoods import Gaussian, Likelihood, NonConjugateLikelihoods
from .means import MeanFunction, Zero
from .types import Array, Dataset


@dataclass
class GP:
    @abstractmethod
    def mean(self) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self) -> Dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def transforms(self) -> Dict:
        raise NotImplementedError


@dataclass(repr=False)
class GPrior(GP):
    kernel: Kernel
    mean_function: Optional[MeanFunction] = Zero()
    name: Optional[str] = "GPrior"

    # def __mul__(self, other: Likelihood):
    #     return construct_posterior(prior=self, likelihood=other)
    def __post_init__(self):
        self._params = {
            "kernel": self.kernel.params,
            "mean_function": self.mean_function.params,
        }

    def mean(self, params: dict) -> Callable[[Array], Array]:
        def mean_fn(X: Array):
            mu = self.mean_function(X, params["mean_function"])
            return mu

        return mean_fn

    def variance(self, params: dict) -> Callable[[Array], Array]:
        def variance_fn(X: Array):
            Kff = gram(self.kernel, X, params["kernel"])
            jitter_matrix = jnp.eye(X.shape[0]) * 1e-8
            covariance_matrix = Kff + jitter_matrix
            return covariance_matrix

        return variance_fn

    @property
    def params(self) -> dict:
        return self._params

    def random_variable(self, X: Array, params: dict) -> tfd.Distribution:
        N = X.shape[0]
        mu = self.mean(params)(X)
        sigma = self.variance(params)(X)
        sigma += jnp.eye(N) * 1e-8
        return tfd.MultivariateNormalTriL(
            mu.squeeze(), linalg.cholesky(sigma, lower=True)
        )

    @property
    def transforms(self) -> Dict:
        return {
            "kernel": self.kernel.transforms,
            "mean_function": self.mean_function.transforms,
        }
