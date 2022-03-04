from collections import namedtuple
from typing import Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp
from chex import dataclass
from jax.scipy.linalg import cholesky, solve_triangular

from gps import GPrior
from jaxgp.likelihoods import Gaussian, Likelihood

from .abstractions import InducingPoints
from .config import Config, default_jitter
from .divergences import gauss_kl
from .kernels import cross_covariance
from .parameters import build_transforms
from .types import Array, Dataset
from .utils import concat_dictionaries, inducingpoint_wrapper


class SVGP:
    def __init__(
        self,
        gprior: GPrior,
        likelihood: Likelihood,
        inducing_points: InducingPoints,
        num_latent_gps: Optional[int] = 1,
        q_diag: bool = False,
        q_mu: Optional[Array] = None,
        q_sqrt: Optional[Array] = None,
        whiten: bool = True,
    ) -> None:
        """_summary_

        Parameters
        ----------
        gprior : GPrior
            _description_
        likelihood : Likelihood
            _description_
        inducing_points : InducingPoints
            _description_
        num_latent_gps : Optional[int], optional
            The number of latent processes to use, by default 1
        q_diag : bool, optional
            If True, the covariance is approximated by a diagonal matrix, by default False
        q_mu : Optional[Array], optional
            _description_, by default None
        q_sqrt : Optional[Array], optional
            _description_, by default None
        whiten : bool, optional
            _description_, by default True
        """
        self.num_latent_gps = num_latent_gps
        self.gprior = gprior
        self.likelihood = likelihood
        self.whiten = whiten
        self.q_diag = q_diag

        self.inducing_points = inducingpoint_wrapper(inducing_points)
        self.num_inducing = self.inducing_points.num_inducing
        self.q_mu, self.q_sqrt = self._init_variational_parameters(
            q_mu, q_sqrt, q_diag
        )
        self._params = concat_dictionaries(
            self.gprior.params,
            self.inducing_points.params,
            {"likelihood": self.likelihood.params},
            {"q_mu": self.q_mu},
            {"q_sqrt": self.q_sqrt},
        )

        self._transforms = concat_dictionaries(
            self.gprior.transforms,
            self.inducing_points.transforms,
            {"likelihood": self.likelihood.transforms},
            {"q_mu": Config.identity_bijector},
            {
                "q_sqrt": Config.positive_bijector
                if q_diag
                else Config.identity_bijector
            },
        )

    def _init_variational_parameters(
        self,
        q_mu: Optional[Array] = None,
        q_sqrt: Optional[Array] = None,
        q_diag: bool = False,
    ):
        if q_mu is None:
            q_mu = jnp.zeros((self.num_inducing, self.num_latent_gps))
        q_mu = jnp.array(q_mu)

        if q_sqrt is None:
            if q_diag:
                q_sqrt = jnp.ones((self.num_inducing, self.num_latent_gps))
            else:
                q_sqrt = [
                    jnp.eye(self.num_inducing)
                    for _ in range(self.num_latent_gps)
                ]
                q_sqrt = jnp.array(q_sqrt)  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                assert self.num_latent_gps == q_sqrt.shape[1]  # [M, P]
            else:
                assert q_sqrt.ndim == 3
                assert self.num_latent_gps == q_sqrt.shape[0]
                assert self.num_inducing == q_sqrt.shape[1] == q_sqrt.shape[2]
        return q_mu, q_sqrt

    @property
    def params(self) -> Dict:
        return self._params

    @property
    def transforms(self) -> Dict:
        return self._transforms

    def prior_kl(self, params: Dict) -> Array:
        if self.whiten:
            return gauss_kl(params["q_mu"], params["q_sqrt"])
        else:
            K = cross_covariance(
                self.gprior.kernel,
                params["inducing_points"],
                params["inducing_points"],
                params["kernel"],
            )
            K = default_jitter(K)
            return gauss_kl(params["q_mu"], params["q_sqrt"], Kp=K)

    def build_elbo(self, num_data: Optional[int] = None, sign=1.0):
        constrain_trans, unconstrain_trans = build_transforms(self.transforms)

        def elbo(raw_params: Dict, data: Dataset):
            params = constrain_trans(raw_params)
            X, Y = data.X, data.Y
            kl = self.prior_kl(params)
            f_mean, f_var = self.predict_f(params, X, full_cov=False)
            var_exp = self.likelihood.variational_expectation(
                params["likelihood"], f_mean, f_var, Y
            )

            if num_data is not None:
                minibatch_size = X.shape[0]
                scale = num_data / minibatch_size
            else:
                scale = 1.0
            return jnp.sum(var_exp) * scale - kl

    def predict_f(
        self, params: Dict, Xnew: Array, full_cov: Optional[bool] = False
    ):
        return
