from abc import abstractmethod, abstractproperty
from audioop import cross
from tkinter.tix import Tree
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
import jax.scipy.linalg as linalg

from typing import Callable, Optional, Dict

from .types import Array, Dataset
from .likelihoods import Likelihood, Gaussian, NonConjugateLikelihoods
from .gps import GP, GPrior
from .utils import concat_dictionaries
from chex import dataclass
from .parameters import copy_dict_structure
from .kernels import gram, cross_covariance
from .config import default_jitter


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
        X, y = training_data.X, training_data.Y
        sigma = params["likelihood"]["noise"]
        n_train = training_data.n
        # Precompute covariance matrices
        Kff = gram(self.prior.kernel, X, params["kernel"])
        prior_mean = self.prior.mean_function(X, params["mean_function"])
        L = linalg.cho_factor(Kff + jnp.eye(n_train) * sigma, lower=True)

        prior_distance = y - prior_mean
        weights = linalg.cho_solve(L, prior_distance)

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
        L = linalg.cho_factor(Kff + jnp.eye(n_train) * variance, lower=True)

        def variance_fn(test_inputs: Array) -> Array:
            Kfx = cross_covariance(
                self.prior.kernel, X, test_inputs, params["kernel"]
            )
            Kxx = gram(self.prior.kernel, test_inputs, params["kernel"])
            latent_values = linalg.cho_solve(L, Kfx.T)
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
        x, y = training.X, training.Y

        def mll(
            params: dict,
        ):
            params = transform(params=params, transform_map=transformations)
            if static_params:
                params = concat_dictionaries(params, transform(static_params))
            mu = self.prior.mean_function(x, params)
            gram_matrix = gram(self.prior.kernel, x, params["kernel"])
            gram_matrix += params["likelihood"]["noise"] * jnp.eye(x.shape[0])
            L = linalg.cholesky(gram_matrix, lower=True)
            random_variable = tfd.MultivariateNormalTriL(mu, L)

            log_prior_density = evaluate_priors(params, priors)
            constant = jnp.array(-1.0) if negative else jnp.array(1.0)
            return constant * (
                random_variable.log_prob(y.squeeze()).mean()
                + log_prior_density
            )

        return mll


class SGPRPosterior:
    def __init__(self, train_data: Dataset, gprior: GPrior) -> None:
        self.train_data = train_data
        self.gprior = gprior
        self.num_latent_gps = train_data.Y.shape[-1]

    def predict_f(self, X_new: Array, params: Dict, full_cov: bool = False):
        if X_new.ndim == 1:
            X_new = X_new[..., None]
        X, Y = self.train_data.X, self.train_data.Y
        iv = params["inducing_points"]
        num_inducing = iv.shape[0]
        err = Y - self.gprior.mean(params)(X)
        Kuf = cross_covariance(self.gprior.kernel, iv, X, params["kernel"])
        Kuu = cross_covariance(self.gprior.kernel, iv, iv, params["kernel"])
        Kuu = default_jitter(Kuu)
        Kus = cross_covariance(self.gprior.kernel, iv, X_new, params["kernel"])
        sigma = jnp.sqrt(params["likelihood"]["noise"])
        L = linalg.cholesky(Kuu, lower=True)
        A = linalg.solve_triangular(L, Kuf, lower=True) / sigma
        B = A @ A.T + jnp.eye(num_inducing)
        LB = linalg.cholesky(B, lower=True)
        Aerr = A @ err
        c = linalg.solve_triangular(LB, Aerr, lower=True) / sigma
        tmp1 = linalg.solve_triangular(L, Kus, lower=True)
        tmp2 = linalg.solve_triangular(LB, tmp1, lower=True)
        mean = tmp2.T @ c
        if full_cov:
            var = (
                cross_covariance(
                    self.gprior.kernel, X_new, X_new, params["kernel"]
                )
                + tmp2.T @ tmp2
                - tmp1.T @ tmp1
            )
            var = jnp.tile(var[None, ...], [self.num_latent_gps, 1, 1])
        else:
            var = (
                gram(self.gprior.kernel, X_new, params["kernel"], diag=True)
                + jnp.sum(tmp2 ** 2, 0)
                - jnp.sum(tmp1 ** 2, 0)
            )
            var = jnp.tile(var[None, ...], [self.num_latent_gps, 1])
        return mean, var


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
