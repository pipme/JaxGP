import copy
from collections import namedtuple
from typing import Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg

from .abstractions import InducingPoints
from .config import default_jitter
from .gps import GPrior
from .kernels import cross_covariance, gram
from .likelihoods import Gaussian, Likelihood
from .parameters import build_transforms
from .posteriors import SGPRPosterior
from .types import Array, Dataset
from .utils import concat_dictionaries, inducingpoint_wrapper


class SGPR:
    def __init__(
        self,
        train_data: Dataset,
        gprior: GPrior,
        likelihood: Gaussian,
        inducing_points: InducingPoints,
        hyp_prior: Optional[GPrior] = None,
    ) -> None:
        self.train_data = train_data
        self.gprior = gprior
        self.likelihood = likelihood
        self.inducing_points = inducingpoint_wrapper(inducing_points)

        self.hyp_prior = hyp_prior
        self.num_data = self.train_data.Y.shape[0]
        self.num_latent_gps = self.train_data.Y.shape[1]
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

    @property
    def params(self) -> Dict:
        return self._params

    @property
    def transforms(self) -> Dict:
        return self._transforms

    def _common_calculation(self, params: Dict):
        X = self.train_data.X
        iv = params["inducing_points"]
        sigma_sq = params["likelihood"]["noise"]

        Kuf = cross_covariance(self.gprior.kernel, iv, X, params["kernel"])
        Kuu = cross_covariance(self.gprior.kernel, iv, iv, params["kernel"])
        Kuu = default_jitter(Kuu)
        L = linalg.cholesky(Kuu, lower=True)
        sigma = jnp.sqrt(sigma_sq)

        # Compute intermediate matrices
        A = linalg.solve_triangular(L, Kuf, lower=True) / sigma
        AAT = A @ A.T
        B = AAT + jnp.eye(AAT.shape[0])
        LB = linalg.cholesky(B, lower=True)
        return namedtuple("CommonTensors", ["A", "B", "LB", "AAT", "L"])(
            A, B, LB, AAT, L
        )

    def logdet_term(self, params: Dict, common: NamedTuple):
        LB = common.LB
        AAT = common.AAT
        X, Y = self.train_data.X, self.train_data.Y
        num_data = self.num_data
        outdim = Y.shape[1]
        kdiag = gram(self.gprior.kernel, X, params["kernel"], full_cov=False)
        sigma_sq = params["likelihood"]["noise"]

        # tr(K) / sigma^2
        trace_k = jnp.sum(kdiag) / sigma_sq
        # tr(Q) / sigma^2
        trace_q = jnp.trace(AAT)
        # tr(K - Q) / sigma^2
        trace = trace_k - trace_q

        # log(det(B))
        log_det_b = 2 * jnp.sum(jnp.log(jnp.diag(LB)))
        # N * log(sigma^2)
        log_sigma_sq = num_data * jnp.log(sigma_sq)

        logdet_k = -outdim * 0.5 * (log_det_b + log_sigma_sq + trace)
        return logdet_k

    def quad_term(self, params: Dict, common: NamedTuple):
        A = common.A
        LB = common.LB

        X, Y = self.train_data.X, self.train_data.Y
        err = Y - self.gprior.mean(params)(X)
        sigma_sq = params["likelihood"]["noise"]
        sigma = jnp.sqrt(sigma_sq)

        Aerr = A @ err
        c = linalg.solve_triangular(LB, Aerr, lower=True) / sigma

        # sigma^2 * y^T @ y
        err_inner_prod = jnp.sum(err ** 2) / sigma_sq
        c_inner_prod = jnp.sum(c ** 2)

        quad = -0.5 * (err_inner_prod - c_inner_prod)
        return quad

    def build_elbo(self, sign=1.0):
        X, Y = self.train_data.X, self.train_data.Y
        constrain_trans, unconstrain_trans = build_transforms(self.transforms)

        def elbo(raw_params: Dict):
            # transform params to constrained space
            params = constrain_trans(raw_params)
            common = self._common_calculation(params)
            num_data = self.num_data
            output_dim = self.num_latent_gps
            const = -0.5 * num_data * output_dim * jnp.log(2 * jnp.pi)
            logdet = self.logdet_term(params, common)
            quad = self.quad_term(params, common)
            return sign * (const + logdet + quad)

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
        sig = Kuu + (params["likelihood"]["noise"] ** -1) * Kuf @ Kuf.T
        sig_sqrt = linalg.cholesky(sig, lower=True)
        sig_sqrt_kuu = linalg.solve_triangular(sig_sqrt, Kuu, lower=True)

        cov = sig_sqrt_kuu.T @ sig_sqrt_kuu
        err = Y - self.gprior.mean(params)(X)
        mu = (
            sig_sqrt_kuu.T
            @ linalg.solve_triangular(sig_sqrt, Kuf @ err, lower=True)
            / params["likelihood"]["noise"]
        )

        return mu, cov

    def posterior(self):
        return SGPRPosterior(self.train_data, self.gprior)
