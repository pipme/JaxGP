from abc import abstractmethod
from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import jax.scipy.linalg as linalg
import tensorflow_probability.substrates.jax.distributions as tfd

from .config import default_jitter
from .datasets import Dataset
from .gps import GP, GPrior
from .helpers import Array, dataclass
from .kernels import cross_covariance, gram
from .likelihoods import Gaussian, Likelihood, NonConjugateLikelihoods
from .parameters import copy_dict_structure
from .utils import concat_dictionaries


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
class GPRPosterior:
    train_data: Dataset
    gprior: GPrior
    Lchol: Array
    hyp_params: Dict

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_latent_gps", self.train_data.Y.shape[-1])
        if self.num_latent_gps != 1:  # type: ignore
            raise NotImplementedError(
                "Currently only one latent GP is supported for GPR."
            )

    def predict_f(
        self, X_test: Array, full_cov: bool = False
    ) -> Tuple[Array, Array]:
        if X_test.ndim == 1:
            X_test = X_test[..., None]
        X, Y = self.train_data.X, self.train_data.Y
        prior_distance = Y - self.gprior.mean(self.hyp_params)(X)
        weights = linalg.cho_solve((self.Lchol, True), prior_distance)
        prior_mean_at_test_inputs = self.gprior.mean(self.hyp_params)(X_test)
        Ktx = cross_covariance(
            self.gprior.kernel, X_test, X, self.hyp_params["kernel"]
        )

        mean = prior_mean_at_test_inputs + Ktx @ weights

        Ktt = gram(
            self.gprior.kernel, X_test, self.hyp_params["kernel"], full_cov
        )
        tmp = linalg.solve_triangular(self.Lchol, Ktx.T, lower=True)
        if full_cov:
            cov = Ktt - tmp.T @ tmp
        else:
            cov = Ktt - jnp.sum(jnp.square(tmp), 0)
        return mean, cov


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
                gram(
                    self.gprior.kernel, X_new, params["kernel"], full_cov=False
                )
                + jnp.sum(tmp2**2, 0)
                - jnp.sum(tmp1**2, 0)
            )
            var = jnp.tile(var[None, ...], [self.num_latent_gps, 1])
        return mean, var


class HeteroskedasticSGPRPosterior:
    def __init__(
        self,
        train_data: Dataset,
        gprior: GPrior,
        likelihood: Likelihood,
        sigma_sq_user: Array,
    ) -> None:
        self.train_data = train_data
        self.gprior = gprior
        self.likelihood = likelihood
        self.num_latent_gps = train_data.Y.shape[-1]
        self.sigma_sq_user = sigma_sq_user

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
        sigma_sq = self.likelihood.compute(params, self.sigma_sq_user)
        sigma = jnp.sqrt(sigma_sq)
        L = linalg.cholesky(Kuu, lower=True)
        A = linalg.solve_triangular(L, Kuf, lower=True) / sigma[None, :]
        B = A @ A.T + jnp.eye(num_inducing)
        LB = linalg.cholesky(B, lower=True)
        Aerr = (A / sigma[None, :]) @ err
        c = linalg.solve_triangular(LB, Aerr, lower=True)
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
                gram(
                    self.gprior.kernel, X_new, params["kernel"], full_cov=False
                )
                + jnp.sum(tmp2**2, 0)
                - jnp.sum(tmp1**2, 0)
            )
            var = jnp.tile(var[None, ...], [self.num_latent_gps, 1])
        return mean, var
