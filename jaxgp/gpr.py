from typing import Dict, Optional

import jax.numpy as jnp
import jax.scipy.linalg as linalg

from .config import default_jitter
from .datasets import Dataset
from .gps import GPrior
from .helpers import Array
from .kernels import cross_covariance, gram
from .likelihoods import FixedHeteroskedasticGaussian, Gaussian, Likelihood
from .parameters import build_transforms
from .posteriors import GPRPosterior
from .utils import concat_dictionaries


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
        sigma_sq: Optional[Array] = None,
    ) -> None:
        self.train_data = train_data
        self.gprior = gprior
        self.sigma_sq = sigma_sq
        if sigma_sq is not None:
            self.sigma_sq = self.sigma_sq.squeeze()
            self.likelihood = FixedHeteroskedasticGaussian()
        else:
            self.likelihood = Gaussian()
        self.num_latent_gps = self.train_data.Y.shape[1]
        self._params = concat_dictionaries(
            self.gprior.params,
            {"likelihood": self.likelihood.params},
        )
        self._transforms = concat_dictionaries(
            self.gprior.transforms,
            {"likelihood": self.likelihood.transforms},
        )

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
            if self.sigma_sq is not None:
                covariance = Kxx + jnp.diag(self.sigma_sq)
            else:
                covariance = Kxx + params["likelihood"]["noise"] * jnp.eye(
                    X.shape[0]
                )
            covariance = default_jitter(covariance)
            L = linalg.cholesky(covariance, lower=True)
            det_sqrt = jnp.prod(jnp.diag(L))

            mll_value = (
                -0.5
                * jnp.sum((Y - mu) * linalg.cho_solve((L, True), Y - mu), 0)
                - jnp.log(det_sqrt)
                - N / 2 * jnp.log(2 * jnp.pi)
            )  # [N, L]
            mll_value = mll_value.mean()
            # TODO: missing priors for params
            # log_prior_density = evaluate_priors(params, priors)
            return sign * mll_value

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
        if self.sigma_sq is not None:
            covariance = Kxx + jnp.diag(self.sigma_sq)
        else:
            covariance = Kxx + params["likelihood"]["noise"] * jnp.eye(
                X.shape[0]
            )
        covariance = default_jitter(covariance)
        Lchol = linalg.cholesky(covariance, lower=True)
        return GPRPosterior(
            train_data=self.train_data,
            gprior=self.gprior,
            Lchol=Lchol,
            hyp_params=params,
        )
