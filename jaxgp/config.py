from dataclasses import dataclass

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors


@dataclass(frozen=True)
class Config:
    """Immutable object for storing global GPJax settings

    :param int: Integer data type, int32 or int64.
    :param float: Float data type, float32 or float64
    :param jitter: Used to improve stability of badly conditioned matrices.
            Default value is `1e-6`.
    :param positive_bijector: Method for positive bijector, either "softplus" or "exp".
            Default is "softplus".
    :param positive_minimum: Lower bound for the positive transformation.
    """

    int: type = jnp.int64
    float: type = jnp.float64
    jitter: float = 1e-6
    positive_bijector: tfp.bijectors.Bijector = tfb.Softplus()
    identity_bijector: tfp.bijectors.Bijector = tfb.Identity()
    positive_minimum: float = 0.0


def default_jitter(K):
    return K + Config.jitter * jnp.eye(K.shape[-1])
