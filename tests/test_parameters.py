from distutils.command.build import build
from jaxgp.parameters import build_transforms
from jaxgp.config import Config
import jax.numpy as jnp


def test_build_transforms():
    transforms = {"noise": Config.positive_bijector}
    params = {"noise": jnp.array(-0.504)}
    constrain_trans, unconstrain_trans = build_transforms(transforms)
    constrain_params = constrain_trans(params)
    unconstrain_params = unconstrain_trans(constrain_params)
    assert jnp.allclose(params["noise"], unconstrain_params["noise"])
