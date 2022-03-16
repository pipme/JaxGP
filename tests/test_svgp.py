from jax.config import config

config.update("jax_debug_nans", True)
from dataclasses import dataclass

import jax.numpy as jnp
import jaxgp as jgp
import jaxopt
import numpy as np
from jaxgp.sgpr import SGPR


def test_svgp_optimize():
    return
