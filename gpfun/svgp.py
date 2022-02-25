from operator import imod
import jax
import jax.numpy as jnp
from chex import dataclass

from gpjax.likelihoods import Gaussian, Likelihood
from .utils import concat_dictionaries

from typing import Optional, Dict, NamedTuple
from gps import GPrior
from .abstractions import InducingPoints
from collections import namedtuple
from .types import Array, Dataset
from .kernels import cross_covariance
from jax.scipy.linalg import cholesky, solve_triangular


@dataclass
class SVGP:
    gprior: GPrior
    likelihood: Gaussian
    inducing_points: InducingPoints
    hyp_prior: Optional[GPrior] = None

    def __post_init__(self):
        self._params = {}

    @property
    def params(self) -> dict:
        # variational_params = concat_dictionaries(self.inducing_points.params, )

        gp_hyperparams = concat_dictionaries(
            self.gprior.params, {"likelihood": self.likelihood.params}
        )
        # self._params =
        return gp_hyperparams

    @params.setter
    def params(self, value):
        self._params = value
