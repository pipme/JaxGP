# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax
import jax.numpy as jnp

from jaxgp.config import Config
from jaxgp.parameters import build_transforms


def test_build_transforms():
    transforms = {"noise": Config.positive_bijector}
    params = {"noise": jnp.array([-0.504, 0.3])}
    constrain_trans, unconstrain_trans = build_transforms(transforms)
    constrain_params = constrain_trans(params)
    unconstrain_params = unconstrain_trans(constrain_params)
    assert jnp.allclose(params["noise"], unconstrain_params["noise"])


def test_redundant_transforms():
    transforms = {
        "noise": Config.positive_bijector,
        "r": Config.positive_bijector,
    }
    params = {"noise": jnp.array([-0.504, 0.3])}
    constrain_trans, unconstrain_trans = build_transforms(transforms)
    constrain_trans = jax.jit(constrain_trans)
    unconstrain_trans = jax.jit(unconstrain_trans)
    constrain_params = constrain_trans(params)
    unconstrain_params = unconstrain_trans(constrain_params)
    assert jnp.allclose(params["noise"], unconstrain_params["noise"])
