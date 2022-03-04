from jax.config import config

# Enable Floa64 - this is crucial for more stable matrix inversions.
config.update("jax_enable_x64", True)
# Highlight any potentially unintended broadcasting rank promoting ops.
# config.update("jax_numpy_rank_promotion", "warn")

from .gps import GPrior
from .posteriors import construct_posterior
from .kernels import (
    RBF,
)
from .likelihoods import Bernoulli, Gaussian
from .means import Constant, Zero
from .parameters import copy_dict_structure, initialise
from .types import Dataset
from .sgpr import SGPR

__version__ = "0.0.1"
