import copy

from collections import namedtuple
from typing import Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
from jax import lax
from jax.experimental import checkify

from .abstractions import InducingPoints
from .config import default_jitter
from .datasets import Dataset
from .gps import GPrior
from .helpers import Array
from .kernels import cross_covariance, gram
from .likelihoods import (
    FixedHeteroskedasticGaussian,
    HeteroskedasticGaussianVBMC,
)
from .parameters import build_transforms
from .posteriors import HeteroskedasticSGPRPosterior
from .priors import evaluate_priors
from .utils import (
    concat_dictionaries,
    copy_dict_structure,
    deep_update,
    inducingpoint_wrapper,
)


class HeteroskedasticSGPR:
    def __init__(
        self,
        train_data: Dataset,
        gprior: GPrior,
        likelihood: Union[
            FixedHeteroskedasticGaussian, HeteroskedasticGaussianVBMC
        ],
        inducing_points: InducingPoints,
        sigma_sq_user: Optional[Array] = None,
        hyp_prior: Optional[Dict] = None,
    ) -> None:
        self.train_data = train_data
        self.gprior = gprior
        self.likelihood = likelihood

        if sigma_sq_user is not None:
            self.sigma_sq_user = sigma_sq_user.squeeze()  # [N]
        else:
            self.sigma_sq_user = None

        self.inducing_points = inducingpoint_wrapper(inducing_points)

        self._params = concat_dictionaries(
            self.gprior.params,
            {"likelihood": self.likelihood.params},
            self.inducing_points.params,
        )
        self._transforms = concat_dictionaries(
            self.gprior.transforms,
            {"likelihood": self.likelihood.transforms},
            self.inducing_points.transforms,
        )

        if hyp_prior is not None:
            self.hyp_prior = copy_dict_structure(self._params)
            self.hyp_prior = deep_update(self.hyp_prior, hyp_prior)
        else:
            self.hyp_prior = None

    @property
    def params(self) -> Dict:
        return self._params

    @property
    def transforms(self) -> Dict:
        return self._transforms

    def _common_calculation(self, params: Dict):
        X = self.train_data.X
        iv = params["inducing_points"]
        sigma_sq = self.likelihood.compute(
            params["likelihood"], self.sigma_sq_user
        )  # [N,] or [1,]
        Kuf = cross_covariance(self.gprior.kernel, iv, X, params["kernel"])
        Kuu = gram(self.gprior.kernel, iv, params["kernel"])
        Kuu = default_jitter(Kuu, 10**3)
        L = linalg.cholesky(Kuu, lower=True)
        # cond_fun = lambda x: jnp.any(jnp.isnan(x[0])) & (x[1] < 7)
        # body_fun = lambda x: (
        #     linalg.cholesky(default_jitter(Kuu, 10 ** x[1]), lower=True),
        #     x[1] + 1,
        # )
        # L, i = lax.while_loop(cond_fun, body_fun, (L, 1))
        checkify.checkify(jnp.all(jnp.isfinite(L)), "Cholesky failed")
        sigma = jnp.sqrt(sigma_sq)

        # Compute intermediate matrices
        A = linalg.solve_triangular(L, Kuf, lower=True) / sigma[None, :]
        AAT = A @ A.T
        B = AAT + jnp.eye(AAT.shape[0])
        LB = linalg.cholesky(B, lower=True)
        return namedtuple("CommonTensors", ["A", "B", "LB", "AAT", "L"])(
            A,
            B,
            LB,
            AAT,
            L,
        )

    def logdet_term(self, params: Dict, common: NamedTuple):
        LB = common.LB
        AAT = common.AAT
        A = common.A
        X, Y = self.train_data.X, self.train_data.Y
        outdim = Y.shape[1]
        kdiag = gram(self.gprior.kernel, X, params["kernel"], full_cov=False)
        sigma_sq = self.likelihood.compute(
            params["likelihood"], self.sigma_sq_user
        )  # [N,] or [1,]
        if sigma_sq.size == 1:
            # important for correct computation with Gaussian likelihood
            sigma_sq = jnp.tile(sigma_sq, X.shape[0])
        # tr(KD^{-1})
        # trace_k = jnp.sum(kdiag / sigma_sq)
        # # tr(QD^{-1})
        # trace_q = jnp.trace(AAT)
        # # tr((K - Q)D^{-1})
        # trace = trace_k - trace_q

        trace = jnp.sum(kdiag / sigma_sq - jnp.einsum("ij,ji->i", A.T, A))

        # log(det(B))
        log_det_b = 2 * jnp.sum(jnp.log(jnp.diag(LB)))
        # 1/2 * log(|D|)
        log_sigma_sq = jnp.sum(jnp.log(sigma_sq))

        logdet_k = -outdim * 0.5 * (log_det_b + log_sigma_sq + trace)
        return logdet_k

    def quad_term(self, params: Dict, common: NamedTuple):
        A = common.A
        LB = common.LB

        X, Y = self.train_data.X, self.train_data.Y
        err = Y - self.gprior.mean(params)(X)
        sigma_sq = self.likelihood.compute(
            params["likelihood"], self.sigma_sq_user
        )  # [N,] or [1,]
        sigma = jnp.sqrt(sigma_sq)

        Aerr = (A / sigma[None, :]) @ err
        c = linalg.solve_triangular(LB, Aerr, lower=True)

        # y^T D^{-1} y
        err_inner_prod = jnp.sum((err**2) / sigma_sq[:, None])
        c_inner_prod = jnp.sum(c**2)

        quad = -0.5 * (err_inner_prod - c_inner_prod)
        return quad

    def build_elbo(self, sign=1.0, raw_flag=True):
        X, Y = self.train_data.X, self.train_data.Y
        num_data = X.shape[0]
        num_latent_gps = Y.shape[1]
        constrain_trans, unconstrain_trans = build_transforms(self.transforms)

        def elbo(params: Dict):
            common = self._common_calculation(params)
            output_dim = num_latent_gps
            const = -0.5 * num_data * output_dim * jnp.log(2 * jnp.pi)
            logdet = self.logdet_term(params, common)
            quad = self.quad_term(params, common)
            log_prior_density = evaluate_priors(params, self.hyp_prior)
            return sign * (const + logdet + quad + log_prior_density)

        def elbo_raw(raw_params: Dict):
            # transform params to constrained space
            params = constrain_trans(raw_params)
            return elbo(params)

        if raw_flag:
            return elbo_raw
        else:
            return elbo

    def compute_qu(self, params: Dict) -> Tuple[Array, Array]:
        X, Y = self.train_data.X, self.train_data.Y

        Kuf = cross_covariance(
            self.gprior.kernel, params["inducing_points"], X, params["kernel"]
        )
        Kuu = gram(
            self.gprior.kernel, params["inducing_points"], params["kernel"]
        )
        Kuu = default_jitter(Kuu)
        L = linalg.cholesky(Kuu, lower=True)
        sigma_sq = self.likelihood.compute(
            params["likelihood"], self.sigma_sq_user
        )  # [N]

        # More straightforward but less robust
        # sig = Kuu + Kuf / sigma_sq[None, :] @ Kuf.T
        # sig_sqrt = linalg.cholesky(sig, lower=True)

        sigma = jnp.sqrt(sigma_sq)
        A = linalg.solve_triangular(L, Kuf, lower=True) / sigma[None, :]
        AAT = A @ A.T
        B = AAT + jnp.eye(params["inducing_points"].shape[0])
        LB = linalg.cholesky(B, lower=True)
        sig_sqrt = L @ LB
        sig_sqrt_kuu = linalg.solve_triangular(sig_sqrt, Kuu, lower=True)

        cov = sig_sqrt_kuu.T @ sig_sqrt_kuu
        mu_u = self.gprior.mean(params)(params["inducing_points"])
        err = (
            Y
            - self.gprior.mean(params)(X)
            + Kuf.T @ linalg.cho_solve((L, True), mu_u)
        )
        mu = sig_sqrt_kuu.T @ linalg.solve_triangular(
            sig_sqrt, Kuf / sigma_sq[None, :] @ err + mu_u, lower=True
        )

        return mu, cov

    def posterior(self, params: Optional[Dict] = None):
        return HeteroskedasticSGPRPosterior(
            self.train_data,
            self.gprior,
            self.likelihood,
            self.sigma_sq_user,
            params_cache=params,
        )

    
    # def __deepcopy__(self, memo):
    #     cls = self.__class__
    #     obj = cls.__new__(cls)
    #     memo[id(self)] = obj
    #     for k, v in self.__dict__.items():
    #         if k == "hyp_prior":
    #             hyp_prior_copy = {}
    #             for hyp_name, prior in v.items():
    #                 try:
    #                     hyp_prior_copy[hyp_name] = copy.deepcopy(prior)
    #                 except TypeError as e:
    #                     # A special handling for tensorflow_probability distributions
    #                     if (
    #                         str(e)
    #                         != "missing a required argument: 'distribution'"
    #                     ):
    #                         raise
    #                     else:
    #                         hyp_prior_copy[hyp_name] = prior.copy()
    #             setattr(obj, k, hyp_prior_copy)
    #         else:
    #             setattr(obj, k, copy.deepcopy(v, memo))
    #     return obj
