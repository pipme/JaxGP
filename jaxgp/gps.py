from typing import Callable, Dict, Optional
from abc import abstractmethod, abstractproperty

import jax.numpy as jnp
from jax.scipy import linalg
from numpy import isin

import tensorflow_probability.substrates.jax.distributions as tfd
from chex import dataclass
from .types import Array, Dataset
from .kernels import Kernel, cross_covariance, gram
from .means import MeanFunction, Zero
from .likelihoods import Likelihood, Gaussian, NonConjugateLikelihoods


@dataclass
class GP:
    @abstractmethod
    def mean(self) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractproperty
    def params(self) -> Dict:
        raise NotImplementedError


@dataclass(repr=False)
class GPrior(GP):
    kernel: Kernel
    mean_function: Optional[MeanFunction] = Zero()
    name: Optional[str] = "GPrior"

    # def __mul__(self, other: Likelihood):
    #     return construct_posterior(prior=self, likelihood=other)

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
        return {
            "kernel": self.kernel.params,
            "mean_function": self.mean_function.params,
        }

    def random_variable(self, X: Array, params: dict) -> tfd.Distribution:
        N = X.shape[0]
        mu = self.mean(params)(X)
        sigma = self.variance(params)(X)
        sigma += jnp.eye(N) * 1e-8
        return tfd.MultivariateNormalTriL(
            mu.squeeze(), linalg.cholesky(sigma, lower=True)
        )
