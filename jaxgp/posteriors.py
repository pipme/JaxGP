from abc import abstractmethod
from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import jax.scipy.linalg as linalg
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from .config import default_jitter
from .datasets import Dataset
from .gps import GP, GPrior
from .helpers import Array, dataclass
from .kernels import RBF, cross_covariance, gram
from .likelihoods import Gaussian, Likelihood, NonConjugateLikelihoods
from .means import Quadratic, Zero
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


# TODO: convert to frozen dataclass and cache intermidiate values?
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
        sigma_sq = self.likelihood.compute(
            params["likelihood"], self.sigma_sq_user
        )  # [N,] or [1,]
        sigma = jnp.sqrt(sigma_sq)
        L = linalg.cholesky(Kuu, lower=True)
        A = linalg.solve_triangular(L, Kuf, lower=True) / sigma[None, :]
        B = A @ A.T + jnp.eye(num_inducing)
        LB = linalg.cholesky(B, lower=True)
        Aerr = (A / sigma[None, :]) @ err
        c = linalg.solve_triangular(LB, Aerr, lower=True)
        tmp1 = linalg.solve_triangular(L, Kus, lower=True)
        tmp2 = linalg.solve_triangular(LB, tmp1, lower=True)
        mean1 = tmp2.T @ c  # [N, num_latent_gps]
        mu_u = self.gprior.mean(params)(iv)
        mean2 = linalg.solve_triangular(L, mu_u, lower=True)
        mean2 = linalg.cho_solve((LB, True), A @ A.T @ mean2)
        mean2 = linalg.solve_triangular(L, mean2, lower=True, trans=1)
        mean2 = Kus.T @ mean2
        mean3 = -Kus.T @ linalg.cho_solve((L, True), mu_u)
        mean = mean1 + mean2 + mean3 + self.gprior.mean(params)(X_new)

        if full_cov:
            var = (
                cross_covariance(
                    self.gprior.kernel, X_new, X_new, params["kernel"]
                )
                + tmp2.T @ tmp2
                - tmp1.T @ tmp1
            )
            var = jnp.tile(
                var[None, ...], [self.num_latent_gps, 1, 1]
            )  # [num_latent_gps, N, N]
        else:
            var = (
                gram(
                    self.gprior.kernel, X_new, params["kernel"], full_cov=False
                )
                + jnp.sum(tmp2**2, 0)
                - jnp.sum(tmp1**2, 0)
            )
            var = jnp.tile(
                var[None, ...], [self.num_latent_gps, 1]
            )  # [num_latent_gps, N]
        return mean, var

    def quad(
        self, params: Dict, mu: Array, sigma: Array, compute_var: bool = False
    ):
        """
        Bayesian quadrature for SGPR.

        Compute the integral of a function represented by a Gaussian
        Process with respect to a given Gaussian measure.

        Parameters
        ==========
        mu : array_like
            Either a array of shape ``(N, D)`` with each row containing the
            mean of a single Gaussian measure, or a single floating point
            number which is interpreted as an array of shape ``(1, D)``.
        sigma : array_like
            Either a array of shape ``(N, D)`` with each row containing the
            std of a single Gaussian measure, or a single floating point
            number which is interpreted as an array of shape ``(1, D)``.
        compute_var : bool, defaults to False
            Whether to compute variance for each integral.

        Returns
        =======
        F : ndarray
            The conputed integrals in an array with shape ``(N, 1)``.
        F_var : ndarray, optional
            The computed variances of the integrals in an array with
            shape ``(N, 1)``.
        """

        if not isinstance(self.gprior.kernel, RBF):
            raise ValueError(
                "Bayesian quadrature only supports the squared exponential "
                "kernel."
            )

        X, Y = self.train_data.X, self.train_data.Y
        N, D = X.shape

        if jnp.size(mu) == 1:
            mu = jnp.tile(mu, (1, D))

        N_star = mu.shape[0]
        if jnp.size(sigma) == 1:
            sigma = jnp.tile(sigma, (1, D))

        quadratic_mean_fun = isinstance(self.gprior.mean_function, Quadratic)

        # GP mean function hyperparameters
        if isinstance(self.gprior.mean_function, Zero):
            m0 = 0
        else:
            m0 = params["mean_function"]["mean_const"]

        if quadratic_mean_fun:
            xm = params["mean_function"]["xm"]
            scale = params["mean_function"]["scale"]

        sigma_sq_obs = self.likelihood.compute(params, self.sigma_sq_user)
        sigma_obs = jnp.sqrt(sigma_sq_obs)

        iv = params["inducing_points"]  # [N_iv, D]
        num_inducing = iv.shape[0]

        Kuf = cross_covariance(self.gprior.kernel, iv, X, params["kernel"])
        Kuu = cross_covariance(self.gprior.kernel, iv, iv, params["kernel"])
        Kuu = default_jitter(Kuu)
        L = linalg.cholesky(Kuu, lower=True)
        A = linalg.solve_triangular(L, Kuf, lower=True) / sigma_obs[None, :]
        B = A @ A.T + jnp.eye(num_inducing)
        LB = linalg.cholesky(B, lower=True)

        ell = params["kernel"]["lengthscale"]  # [D]
        sf2 = params["kernel"]["outputscale"]
        tau = jnp.sqrt(sigma**2 + ell**2)  # [N_star, D]
        nf = (
            sf2 * jnp.prod(ell) / jnp.prod(tau, 1)
        )  # Covariance normalization factor, [N_star]

        sum_delta2 = jnp.zeros((N_star, num_inducing))
        for i in range(0, D):
            sum_delta2 += (
                (mu[:, i] - jnp.reshape(iv[:, i], (-1, 1))).T
                / tau[:, i : i + 1]
            ) ** 2
        w = jnp.reshape(nf, (-1, 1)) * jnp.exp(
            -0.5 * sum_delta2
        )  # [N_star, N_iv]

        err = Y - self.gprior.mean(params)(X)  # [N, D]
        Aerr = (A / sigma_obs[None, :]) @ err
        tmp = linalg.cho_solve((LB, True), Aerr)
        tmp = linalg.solve_triangular(L, tmp, lower=True)  # [N_iv, 1]
        F = w @ tmp + m0

        if quadratic_mean_fun:
            nu_k = -0.5 * jnp.sum(
                1
                / scale**2
                * (mu**2 + sigma**2 - 2 * mu * xm + xm**2),
                1,
            )
            F += nu_k

        if compute_var:
            tau_kk = jnp.sqrt(2 * sigma**2 + ell**2)  # [N, D]
            nf_kk = sf2 * jnp.prod(ell) / jnp.prod(tau_kk, 1)  # [N]

            # K_tilde^{-1} = L^-T B^-1 L^-1
            invKwk_1 = linalg.cho_solve((L, True), w.T)
            tmp = linalg.solve_triangular(L, w.T, lower=True)
            tmp = linalg.cho_solve((LB, True), tmp)
            invKwk_2 = linalg.solve_triangular(L.T, tmp)
            invKwk = invKwk_1 - invKwk_2  # [N_iv, N]
            J_kk = nf_kk - jnp.sum(w * invKwk.T, 1)

            F_var = jnp.maximum(jnp.finfo(jnp.float64).eps, J_kk)

        if compute_var:
            return F, F_var

        return F

    def quad_mixture(
        self,
        params: Dict,
        mu: Array,
        sigma: Array,
        weights: Array,
        compute_var: bool = False,
        separate_K: bool = False,
    ):
        """
        Bayesian quadrature for SGPR. (wrt. a Gaussian mixture)

        Compute the integral of a function represented by a Gaussian
        Process with respect to a given Gaussian measure mixture.

        Parameters
        ==========
        mu : array_like
            Either a array of shape ``(K, D)`` with each row containing the
            mean of a single Gaussian measure, or a single floating point
            number which is interpreted as an array of shape ``(1, D)``.
        sigma : array_like
            Either a array of shape ``(K, D)`` with each row containing the
            std of a single Gaussian measure, or a single floating point
            number which is interpreted as an array of shape ``(1, D)``.
        weights: Array
            Weights of the Gaussian measures, of shape (K, )
        compute_var : bool, defaults to False
            Whether to compute variance for each integral.
        separate_K : bool, defaults to False
            Whether to return expected log joint per component.

        Returns
        =======
        F : Array
            The computed integral value.
        F_var : Array or None
            The computed variance of the integral. `F_var` = None if `compute_var` is False.
        I: np.ndarray, optional
            Integral value components with shape ``(K)``.
        J: np.ndarray, optional
            Integral variance components with shape ``(K, K)``.
        """
        if not isinstance(self.gprior.kernel, RBF):
            raise ValueError(
                "Bayesian quadrature only supports the squared exponential "
                "kernel."
            )

        X, Y = self.train_data.X, self.train_data.Y
        N, D = X.shape

        if jnp.size(mu) == 1:
            mu = jnp.tile(mu, (1, D))

        K = mu.shape[0]
        if jnp.size(sigma) == 1:
            sigma = jnp.tile(sigma, (1, D))

        quadratic_mean_fun = isinstance(self.gprior.mean_function, Quadratic)

        # GP mean function hyperparameters
        if isinstance(self.gprior.mean_function, Zero):
            m0 = 0
        else:
            m0 = params["mean_function"]["mean_const"]

        if quadratic_mean_fun:
            xm = params["mean_function"]["xm"]
            scale = params["mean_function"]["scale"]

        sigma_sq_obs = self.likelihood.compute(
            params["likelihood"], self.sigma_sq_user
        )
        sigma_obs = jnp.sqrt(sigma_sq_obs)

        iv = params["inducing_points"]  # [N_iv, D]
        num_inducing = iv.shape[0]

        Kuf = cross_covariance(self.gprior.kernel, iv, X, params["kernel"])
        Kuu = cross_covariance(self.gprior.kernel, iv, iv, params["kernel"])
        Kuu = default_jitter(Kuu)
        L = linalg.cholesky(Kuu, lower=True)
        A = linalg.solve_triangular(L, Kuf, lower=True) / sigma_obs[None, :]
        B = A @ A.T + jnp.eye(num_inducing)
        LB = linalg.cholesky(B, lower=True)

        ell = params["kernel"]["lengthscale"]  # [D]
        sf2 = params["kernel"]["outputscale"].squeeze()
        tau = jnp.sqrt(sigma**2 + ell**2)  # [K, D]
        nf = (
            sf2 * jnp.prod(ell) / jnp.prod(tau, 1)
        )  # Covariance normalization factor, [K]

        sum_delta2 = jnp.zeros((K, num_inducing))
        for i in range(0, D):
            sum_delta2 += (
                (mu[:, i] - jnp.reshape(iv[:, i], (-1, 1))).T
                / tau[:, i : i + 1]
            ) ** 2
        w = jnp.reshape(nf, (-1, 1)) * jnp.exp(-0.5 * sum_delta2)  # [K, N_iv]

        err = Y - self.gprior.mean(params)(X)  # [N, D]
        Aerr = (A / sigma_obs[None, :]) @ err
        tmp = linalg.cho_solve((LB, True), Aerr)
        tmp = linalg.solve_triangular(L, tmp, lower=True)  # [N_iv, 1]
        I = w @ tmp + m0  # [K, 1]
        I = I.squeeze()

        if quadratic_mean_fun:
            nu_k = -0.5 * jnp.sum(
                1
                / scale**2
                * (mu**2 + sigma**2 - 2 * mu * xm + xm**2),
                1,
            )  # [K]
            I += nu_k
        F = jnp.sum(weights * I)

        J = jnp.zeros((K, K))
        F_var = 0.0
        for k in range(K):
            for j in range(k + 1):
                tau_jk = jnp.sqrt(
                    sigma[j, :] ** 2 + sigma[k, :] ** 2 + ell**2
                )  # [D]
                nf_jk = sf2 * jnp.prod(ell) / jnp.prod(tau_jk)
                delta_jk = (mu[j, :] - mu[k, :]) / tau_jk
                J_jk = nf_jk * jnp.exp(-0.5 * jnp.sum(delta_jk**2))

                # K_tilde^{-1} = L^-T B^-1 L^-1
                invKwk_1 = linalg.cho_solve((L, True), w[k])
                tmp = linalg.solve_triangular(L, w[k], lower=True)
                tmp = linalg.cho_solve((LB, True), tmp)
                invKwk_2 = linalg.solve_triangular(L.T, tmp)
                invKwk = invKwk_1 - invKwk_2  # [N_iv, 1]
                invKwk = invKwk.squeeze()
                J_jk -= jnp.dot(w[j], invKwk)

                # Off-diagonal elements are symmetric (count twice)
                if j == k:
                    F_var += (
                        weights[k] ** 2
                        * J_jk
                        * jnp.maximum(jnp.finfo(jnp.float64).eps, J_jk)
                    )
                    if separate_K:
                        J.at[k, k].set(J_jk)
                else:
                    F_var += 2 * weights[j] * weights[k] * J_jk
                    if separate_K:
                        J.at[j, k].set(J_jk)
                        J.at[k, j].set(J_jk)

        # Correct for numerical error
        if compute_var:
            F_var = jnp.maximum(F_var, jnp.finfo(jnp.float64).eps)
        else:
            F_var = None

        if separate_K:
            return F, F_var, I, J
        return F, F_var
