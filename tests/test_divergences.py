from jax.config import config

config.update("jax_debug_nans", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from jaxgp.divergences import gauss_kl, single_gauss_kl


def test_single_gauss_kl():
    keys = jr.split(jr.PRNGKey(42), 2)
    M = 10
    q_mu = jnp.zeros(M)
    q_sqrt = jnp.tril(jr.normal(keys[0], (M, M)))
    Kp = q_sqrt @ q_sqrt.T
    kl_tf = single_gauss_kl(q_mu, q_sqrt, Kp)
    np.testing.assert_allclose(kl_tf, 0.0, atol=1e-3)

    q_sqrt = jnp.abs(jr.normal(keys[1], (M,)))
    kl1 = single_gauss_kl(q_mu, q_sqrt)
    q_sqrt = jnp.tril(jnp.diag(q_sqrt))
    kl2 = single_gauss_kl(q_mu, q_sqrt)
    np.testing.assert_allclose(kl1, kl2, atol=1e-3)


def test_gauss_kl():
    keys = jr.split(jr.PRNGKey(42), 2)
    output_dim = 2
    M = 10
    q_mu = jr.normal(keys[0], (M, output_dim))
    q_sqrt = jr.normal(keys[1], (M, output_dim))
    kl_sum_1 = gauss_kl(q_mu, q_sqrt, None)

    q_sqrt = [jnp.tril(jnp.diag(q_sqrt[:, i])) for i in range(output_dim)]
    q_sqrt = jnp.stack(q_sqrt)
    kl_sum_2 = gauss_kl(q_mu, q_sqrt, None)
    np.testing.assert_allclose(kl_sum_1, kl_sum_2, atol=1e-3)
