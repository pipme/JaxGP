# -*- coding: utf-8 -*-
# mypy: ignore-errors

from jax.config import config

config.update("jax_debug_nans", True)

import jax.numpy as jnp
import jaxopt
import numpy as np

import jaxgp as jgp
from jaxgp.sgpr import SGPR


def test_svgp_optimize():
    return
