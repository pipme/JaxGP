from email.mime import base

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jaxgp.conditionals import base_conditional_with_lm


@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("q_diag", [True, False])
def test_base_conditional_with_lm(full_cov, q_diag):
    M = 10
    N = 25
    keys = jr.split(jr.PRNGKey(42), 5)
    Kmn = jr.normal(keys[0], (M, N))
    Lm = jr.normal(keys[1], (M, M))
    f = jr.normal(keys[2], (M,))
    Knn = jr.normal(keys[3], (N, N))
    if q_diag:
        q_sqrt = jr.normal(keys[4], (M,))
    else:
        q_sqrt = jr.normal(keys[4], (M, M))
    whiten = True
    fmean, fvar = base_conditional_with_lm(
        Kmn, Lm, Knn, f, full_cov, q_sqrt, whiten
    )
    assert fmean.shape == (N,)
    if full_cov:
        assert fvar.shape == (N, N)
    else:
        assert fvar.shape == (N,)
