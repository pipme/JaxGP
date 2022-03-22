import abc
from typing import Callable, Dict, Optional

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from .config import Config
from .helpers import Array, dataclass
from .quadratures import gauss_hermite_quadrature


@dataclass
class Likelihood:
    name: Optional[str] = "Likelihood"

    def __repr__(self) -> str:
        return f"{self.name} likelihood function"

    @property
    @abc.abstractmethod
    def params(self) -> dict:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def link_function(self) -> Callable:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def transforms(self) -> Dict:
        raise NotImplementedError

    def variational_expectation(self, params: Dict, Fmu, Fvar, Y):
        """
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values,

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           ∫ log(p(y=Y|f)) q(f) df.

        This only works if the broadcasting dimension of the statistics of q(f) (mean and variance)
        are broadcastable with that of the data Y.

        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: expected log density of the data given q(F), with shape [...]
        """
        raise NotImplementedError


@dataclass
class Gaussian(Likelihood):
    name: Optional[str] = "Gaussian"

    @property
    def params(self) -> Dict:
        return {"noise": jnp.array(1.0)}

    @property
    def link_function(self) -> Callable:
        def identity_fn(x):
            return x

        return identity_fn

    @property
    def transforms(self) -> Dict:
        return {"noise": Config.positive_bijector}

    def variational_expectation(
        self, params: Dict, Fmu: Array, Fvar: Array, Y: Array
    ):
        sigma_sq = params["noise"]
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * jnp.log(sigma_sq)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / sigma_sq,
            axis=-1,
        )

    def predict_mean_and_var(self, params: Dict, Fmu, Fvar):
        return Fmu, Fvar + params["noise"]


@dataclass
class Bernoulli(Likelihood):
    name: Optional[str] = "Bernoulli"

    @property
    def params(self) -> dict:
        return {}

    @property
    def link_function(self) -> Callable:
        def link_fn(x):
            return tfd.ProbitBernoulli(x)

        return link_fn

    @property
    def predictive_moment_fn(self) -> Callable:
        def moment_fn(mean: Array, variance: Array):
            rv = self.link_function(mean / jnp.sqrt(1 + variance))
            return rv

        return moment_fn

    def _log_prob(self, F: Array, Y: Array) -> Array:
        """Compute log probabilities.

        Parameters
        ----------
        F : Array
            Latent function values.
        Y : Array
            Observations.

        Returns
        -------
        Array
            Log probabilities.
        """
        return self.link_function(F).log_prob(Y)

    @property
    def transforms(self) -> Dict:
        return {}

    def variational_expectation(self, params: Dict, Fmu, Fvar, Y):
        return gauss_hermite_quadrature(self._log_prob, Fmu, Fvar, Y=Y)


NonConjugateLikelihoods = [Bernoulli]
NonConjugateLikelihoodType = Bernoulli  # Union[Bernoulli]
