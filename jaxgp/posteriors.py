from abc import abstractmethod, abstractproperty
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from typing import Callable, Optional, Dict

from .types import Array, Dataset
from .likelihoods import Likelihood, Gaussian, NonConjugateLikelihoods
from .gps import GP, GPrior
from .utils import concat_dictionaries
from chex import dataclass
from .parameters import copy_dict_structure
from jax.scipy.linalg import cho_factor, cho_solve, cholesky
from .kernels import gram, cross_covariance


@dataclass
class Posterior(GP):
    prior: GPrior
    likelihood: Likelihood
    name: Optional[str] = "GP Posterior"

    @abstractmethod
    def mean(
        self, training_data: Dataset, params: dict
    ) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractmethod
    def variance(
        self, training_data: Dataset, params: dict
    ) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @property
    def params(self) -> dict:
        return concat_dictionaries(
            self.prior.params, {"likelihood": self.likelihood.params}
        )


@dataclass
class ConjugatePosterior(Posterior):
    prior: GPrior
    likelihood: Gaussian
    name: Optional[str] = "ConjugatePosterior"

    def mean(
        self, training_data: Dataset, params: dict
    ) -> Callable[[Array], Array]:
        X, y = training_data.X, training_data.y
        sigma = params["likelihood"]["noise"]
        n_train = training_data.n
        # Precompute covariance matrices
        Kff = gram(self.prior.kernel, X, params["kernel"])
        prior_mean = self.prior.mean_function(X, params["mean_function"])
        L = cho_factor(Kff + jnp.eye(n_train) * sigma, lower=True)

        prior_distance = y - prior_mean
        weights = cho_solve(L, prior_distance)

        def mean_fn(test_inputs: Array) -> Array:
            prior_mean_at_test_inputs = self.prior.mean_function(
                test_inputs, params["mean_function"]
            )
            Kfx = cross_covariance(
                self.prior.kernel, X, test_inputs, params["kernel"]
            )
            return prior_mean_at_test_inputs + jnp.dot(Kfx, weights)

        return mean_fn

    def variance(
        self, training_data: Dataset, params: dict
    ) -> Callable[[Array], Array]:
        X = training_data.X
        n_train = training_data.n
        variance = params["likelihood"]["noise"]
        n_train = training_data.n
        Kff = gram(self.prior.kernel, X, params["kernel"])
        Kff += jnp.eye(n_train) * 1e-8
        L = cho_factor(Kff + jnp.eye(n_train) * variance, lower=True)

        def variance_fn(test_inputs: Array) -> Array:
            Kfx = cross_covariance(
                self.prior.kernel, X, test_inputs, params["kernel"]
            )
            Kxx = gram(self.prior.kernel, test_inputs, params["kernel"])
            latent_values = cho_solve(L, Kfx.T)
            return Kxx - jnp.dot(Kfx, latent_values)

        return variance_fn

    def marginal_log_likelihood(
        self,
        training: Dataset,
        transformations: Dict,
        priors: dict = None,
        static_params: dict = None,
        negative: bool = False,
    ) -> Callable[[Dataset], Array]:
        x, y = training.X, training.y

        def mll(
            params: dict,
        ):
            params = transform(params=params, transform_map=transformations)
            if static_params:
                params = concat_dictionaries(params, transform(static_params))
            mu = self.prior.mean_function(x, params)
            gram_matrix = gram(self.prior.kernel, x, params["kernel"])
            gram_matrix += params["likelihood"]["noise"] * jnp.eye(x.shape[0])
            L = cholesky(gram_matrix, lower=True)
            random_variable = tfd.MultivariateNormalTriL(mu, L)

            log_prior_density = evaluate_priors(params, priors)
            constant = jnp.array(-1.0) if negative else jnp.array(1.0)
            return constant * (
                random_variable.log_prob(y.squeeze()).mean()
                + log_prior_density
            )

        return mll


def construct_posterior(
    prior: GPrior, likelihood: Likelihood, method: str = "exact"
) -> Posterior:
    if method == "exact":
        assert isinstance(likelihood, Gaussian)
        PosteriorGP = ConjugatePosterior
    else:
        raise NotImplementedError(
            f"No posterior implemented for {likelihood.name} likelihood"
        )
    return PosteriorGP(prior=prior, likelihood=likelihood)
