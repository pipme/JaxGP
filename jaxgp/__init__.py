from jax.config import config

# Enable Floa64 - this is crucial for more stable matrix inversions.
config.update("jax_enable_x64", True)
# Highlight any potentially unintended broadcasting rank promoting ops.
# config.update("jax_numpy_rank_promotion", "warn")

from jaxgp import gps, kernels, likelihoods, means
from jaxgp.config import Config
from jaxgp.datasets import Dataset
from jaxgp.gps import GPrior
from jaxgp.parameters import copy_dict_structure, initialise
from jaxgp.sgpr import SGPR
from jaxgp.sgpr_heteroskedastic import HeteroskedasticSGPR
from jaxgp.svgp import SVGP

__author__ = "Chengkun Li"
__email__ = "sjtulck@gmail.com"
__uri__ = "https://github.com/pipme/JaxGP"
__copyright__ = "Copyright 2022, Machine and Human Intelligence Group, University of Helsinki"
__license__ = "MIT"
__description__ = "A Gaussian Process library in Jax"
__version__ = "0.0.1"
