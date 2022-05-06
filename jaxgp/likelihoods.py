import abc
from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from .config import Config
from .helpers import Array, dataclass
from .quadratures import gauss_hermite_quadrature


@dataclass
class Likelihood:
    name: str = ""

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

    def compute(self, params: Dict, sigma_sq: Optional[Array] = None):
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
    name: str = "Gaussian"

    @property
    def params(self) -> Dict:
        return {"noise": jnp.array([1.0])}

    @property
    def link_function(self) -> Callable:
        def identity_fn(x):  # type: ignore
            return x

        return identity_fn

    @property
    def transforms(self) -> Dict:
        return {"noise": Config.positive_bijector}

    def compute(self, params: Dict, *args):
        return params["noise"]

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

    def predict_mean_and_var(
        self, params: Dict, Fmu: Array, Fvar: Array, full_cov: bool = False
    ) -> Tuple[Array, Array]:
        """Predict mean and var/cov for y.

        Parameters
        ----------
        params : Dict
            Parameter dictionary.
        Fmu : Array
            Mean values, with shape [N, latent_dim].
        Fvar : Array
            Variance or covariance matrix values, with shape [N, latent_dim]
            or [latent_dim, N, N].

        full_cov : bool, optional
            Whether to compute covariance matrix, defalut=False.

        Returns
        -------
        Ymu: Array
            Mean values.
        Yvar: Array
            Variance or covariance matrix values.
        """
        if full_cov:
            assert Fvar.ndim >= 3
            # For details, see discussions in https://github.com/google/jax/issues/2680#issuecomment-804269672
            i, j = jnp.diag_indices(min(Fvar.shape[-2:]))
            Ymu = Fmu
            Yvar = Fvar.at[..., i, j].add(params["noise"])
            return Ymu, Yvar
        else:
            assert Fvar.ndim == 2
            Ymu = Fmu
            Yvar = Fvar + params["noise"]
        return Ymu, Yvar


@dataclass
class FixedHeteroskedasticGaussian(Likelihood):
    """Fixed heteroskedastic Gaussian noise."""

    name: str = "Fixed Heteroskedastic Gaussian"

    @property
    def params(self) -> Dict:
        return {}

    @property
    def link_function(self) -> Callable:
        def identity_fn(x):  # type: ignore
            return x

        return identity_fn

    @property
    def transforms(self) -> Dict:
        return {}

    def compute(self, params: Dict, sigma_sq: Optional[Array] = None):
        if sigma_sq is None:
            return jnp.array([Config.jitter**2])
        return sigma_sq.squeeze()

    def variational_expectation(
        self,
        params: Dict,
        Fmu: Array,
        Fvar: Array,
        Y: Array,
        sigma_sq: Optional[Array] = None,
    ) -> Array:
        sigma_sq = self.check_user_provided(params, sigma_sq)
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * jnp.log(sigma_sq)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / sigma_sq,
            axis=-1,
        )

    def predict_mean_and_var(
        self,
        params: Dict,
        Fmu: Array,
        Fvar: Array,
        full_cov: bool = False,
        sigma_sq: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Predict mean and var/cov for y.

        Parameters
        ----------
        params : Dict
            Parameter dictionary.
        Fmu : Array
            Mean values, with shape [N, latent_dim].
        Fvar : Array
            Variance or covariance matrix values, with shape [N, latent_dim]
            or [latent_dim, N, N].

        full_cov : bool, optional
            Whether to compute covariance matrix, defalut=False.

        Returns
        -------
        Ymu: Array
            Mean values.
        Yvar: Array
            Variance or covariance matrix values.
        """
        sigma_sq = self.check_user_provided(params, sigma_sq)
        if full_cov:
            assert Fvar.ndim >= 3
            # For details, see discussions in https://github.com/google/jax/issues/2680#issuecomment-804269672
            i, j = jnp.diag_indices(min(Fvar.shape[-2:]))
            Ymu = Fmu
            sigma_sq = jnp.transpose(sigma_sq)  # [..., N]
            Yvar = Fvar.at[..., i, j].add(sigma_sq)
            return Ymu, Yvar
        else:
            assert Fvar.ndim == 2
            Ymu = Fmu
            Yvar = Fvar + sigma_sq
        return Ymu, Yvar

    def check_user_provided(self, params, sigma_sq):  # type: ignore
        if sigma_sq is None:
            raise ValueError("sigma_sq should be provided")
        assert jnp.all(sigma_sq >= 0)

        if sigma_sq.ndim == 1:
            sigma_sq = jnp.expand_dims(sigma_sq, -1)  # [N, 1]
        assert sigma_sq.ndim == 2
        return sigma_sq  # [N, 1] or [N, latent_dim]


@dataclass
class HeteroskedasticGaussianVBMC(Likelihood):
    """_summary_

    Parameters
    ----------
    constant_add : bool, defaults to False
        Whether to add constant noise.
    user_provided_add : bool, defaults to False
        Whether to add user provided (input) noise.
    scale_user_provided : bool, defaults to False
        Whether to scale uncertainty in provided noise. If
        ``user_provided_add = False`` then this does nothing.
    rectified_linear_output_dependent_add : bool, defaults to False
        Whether to add rectified linear output-dependent noise.
    """

    constant_add: bool = False
    user_provided_add: bool = False
    scale_user_provided: bool = False
    rectified_linear_output_dependent_add: bool = False
    name: str = "Heteroskedastic Gaussian for VBMC"

    def __post_init__(self) -> None:
        if self.rectified_linear_output_dependent_add:
            raise NotImplementedError

    @property
    def params(self) -> Dict:
        params = {}
        if self.constant_add:
            params["noise_add"] = jnp.array([1.0])
        if self.user_provided_add and self.scale_user_provided:
            params["scale_user_noise"] = jnp.array([1.0])
        if self.rectified_linear_output_dependent_add:
            params["intercept_rectify"] = jnp.array([0.0])
            params["scale_rectify"] = jnp.array([1.0])
        return params

    @property
    def link_function(self) -> Callable:
        def identity_fn(x):  # type: ignore
            return x

        return identity_fn

    @property
    def transforms(self) -> Dict:
        transforms = {}
        if self.constant_add:
            transforms["noise_add"] = Config.positive_bijector
        if self.user_provided_add and self.scale_user_provided:
            transforms["scale_user_noise"] = Config.positive_bijector
        if self.rectified_linear_output_dependent_add:
            transforms["intercept_rectify"] = Config.identity_bijector
            transforms["scale_rectify"] = Config.positive_bijector
        return transforms

    def compute(self, params: Dict, sigma_sq: Optional[Array] = None) -> Array:
        if sigma_sq is not None:
            sigma_sq = sigma_sq.squeeze()
            assert sigma_sq.ndim == 1
        if self.constant_add:
            noise_var = jnp.array(params["noise_add"])
        else:
            noise_var = jnp.finfo(jnp.float64).eps

        if self.user_provided_add:
            assert sigma_sq is not None, "sigma_sq need to be provided"
            noise_var += sigma_sq
        elif self.scale_user_provided:
            assert sigma_sq is not None, "sigma_sq need to be provided"
            noise_var += params["scale_user"] * sigma_sq

        return noise_var

    def variational_expectation(
        self,
        params: Dict,
        Fmu: Array,
        Fvar: Array,
        Y: Array,
        sigma_sq: Optional[Array] = None,
    ) -> Array:
        sigma_sq = self.compute(params, sigma_sq)
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * jnp.log(sigma_sq)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / sigma_sq,
            axis=-1,
        )

    def predict_mean_and_var(
        self,
        params: Dict,
        Fmu: Array,
        Fvar: Array,
        full_cov: bool = False,
        sigma_sq: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Predict mean and var/cov for y.

        Parameters
        ----------
        params : Dict
            Parameter dictionary.
        Fmu : Array
            Mean values, with shape [N, latent_dim].
        Fvar : Array
            Variance or covariance matrix values, with shape [N, latent_dim]
            or [latent_dim, N, N].

        full_cov : bool, optional
            Whether to compute covariance matrix, defalut=False.

        Returns
        -------
        Ymu: Array
            Mean values.
        Yvar: Array
            Variance or covariance matrix values.
        """
        sigma_sq = self.compute(params, sigma_sq)
        if full_cov:
            assert Fvar.ndim >= 3
            # For details, see discussions in https://github.com/google/jax/issues/2680#issuecomment-804269672
            i, j = jnp.diag_indices(min(Fvar.shape[-2:]))
            Ymu = Fmu
            sigma_sq = jnp.transpose(sigma_sq)  # [..., N]
            Yvar = Fvar.at[..., i, j].add(sigma_sq)
            return Ymu, Yvar
        else:
            assert Fvar.ndim == 2
            Ymu = Fmu
            Yvar = Fvar + sigma_sq
        return Ymu, Yvar


@dataclass
class Bernoulli(Likelihood):
    name: str = "Bernoulli"

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
