from typing import Dict, Optional, Union

import jax.numpy as jnp
import jax.scipy.linalg as linalg

from .config import default_jitter
from .datasets import Dataset
from .gps import GPrior
from .helpers import Array
from .kernels import cross_covariance, gram
from .likelihoods import (
    FixedHeteroskedasticGaussian,
    Gaussian,
    HeteroskedasticGaussianVBMC,
    Likelihood,
)
from .parameters import build_transforms
from .posteriors import GPRPosterior
from .priors import evaluate_priors
from .utils import concat_dictionaries, copy_dict_structure, deep_update


class GPR:
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is given by

    .. math::
       \log p(Y \,|\, \mathbf f) =
            \mathcal N(Y \,|\, 0, \sigma_n^2 \mathbf{I})

    To train the model, we maximise the log _marginal_ likelihood
    w.r.t. the likelihood variance and kernel hyperparameters theta.
    The marginal likelihood is found by integrating the likelihood
    over the prior, and has the form

    .. math::
       \log p(Y \,|\, \sigma_n, \theta) =
            \mathcal N(Y \,|\, 0, \mathbf{K} + \sigma_n^2 \mathbf{I})
    """

    def __init__(
        self,
        train_data: Dataset,
        gprior: GPrior,
        likelihood: Union[
            Gaussian, FixedHeteroskedasticGaussian, HeteroskedasticGaussianVBMC
        ],
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

        self.num_latent_gps = self.train_data.Y.shape[1]
        self._params = concat_dictionaries(
            self.gprior.params,
            {"likelihood": self.likelihood.params},
        )
        self._transforms = concat_dictionaries(
            self.gprior.transforms,
            {"likelihood": self.likelihood.transforms},
        )
        if hyp_prior is not None:
            self.hyp_prior = copy_dict_structure(self._params)
            self.hyp_prior = deep_update(self.hyp_prior, hyp_prior)
        else:
            self.hyp_prior = None

    def build_mll(
        self, static_params: Optional[Dict] = None, sign: float = 1.0
    ):
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.train_data.X, self.train_data.Y
        N = X.shape[0]
        constrain_trans, unconstrain_trans = build_transforms(self.transforms)

        def mll(
            raw_params: Dict,
        ):
            # transform params to constrained space
            params = constrain_trans(raw_params)
            # TODO: better way?
            if static_params:
                params = concat_dictionaries(params, static_params)

            mu = self.gprior.mean(params)(X)
            Kxx = gram(self.gprior.kernel, X, params["kernel"])
            sigma_sq = self.likelihood.compute(
                params["likelihood"], self.sigma_sq_user
            )
            if sigma_sq.shape[0] == 1:
                sigma_sq = jnp.repeat(sigma_sq, Kxx.shape[0])
            covariance = Kxx + jnp.diag(sigma_sq)
            covariance = default_jitter(covariance)
            L = linalg.cholesky(covariance, lower=True)
            log_det_sqrt = jnp.sum(jnp.log(jnp.diag(L)))
            mll_value = (
                -0.5
                * jnp.sum((Y - mu) * linalg.cho_solve((L, True), Y - mu), 0)
                - log_det_sqrt
                - N / 2 * jnp.log(2 * jnp.pi)
            )  # [N, L]
            mll_value = mll_value.mean()
            log_prior_density = evaluate_priors(params, self.hyp_prior)
            return sign * (mll_value + log_prior_density)

        return mll

    @property
    def params(self) -> Dict:
        return self._params

    @property
    def transforms(self) -> Dict:
        return self._transforms

    def posterior(self, params: Dict):
        X = self.train_data.X
        Kxx = gram(self.gprior.kernel, X, params["kernel"])
        sigma_sq = self.likelihood.compute(
            params["likelihood"], self.sigma_sq_user
        )
        if sigma_sq.shape[0] == 1:
            sigma_sq = jnp.repeat(sigma_sq, Kxx.shape[0])
        covariance = Kxx + jnp.diag(sigma_sq)
        covariance = default_jitter(covariance)
        Lchol = linalg.cholesky(covariance, lower=True)
        return GPRPosterior(
            train_data=self.train_data,
            gprior=self.gprior,
            Lchol=Lchol,
            hyp_params=params,
        )
