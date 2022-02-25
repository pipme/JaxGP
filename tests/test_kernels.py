from itertools import permutations
from chex import assert_equal

import jax.numpy as jnp
import jax.random as jr
from numpy import isin, sort
import pytest
from gpfun import kernels
from gpfun.kernels import (
    RBF,
    cross_covariance,
    gram,
)
from gpfun.parameters import initialise


@pytest.mark.parametrize("kern", [RBF()])
@pytest.mark.parametrize("dim", [1, 2, 5])
def test_gram(kern, dim):
    x = jnp.linspace(-1.0, 1.0, num=10).reshape(-1, 1)
    if dim > 1:
        x = jnp.hstack([x] * dim)
    params, _, _ = initialise(kern)
    gram_matrix = gram(kern, x, params)
    assert gram_matrix.shape[0] == x.shape[0]
    assert gram_matrix.shape[0] == gram_matrix.shape[1]


@pytest.mark.parametrize("kern", [RBF()])
@pytest.mark.parametrize("n1", [3, 10, 20])
@pytest.mark.parametrize("n2", [3, 10, 20])
def test_cross_covariance(kern, n1, n2):
    x1 = jnp.linspace(-1.0, 1.0, num=n1).reshape(-1, 1)
    x2 = jnp.linspace(-1.0, 1.0, num=n2).reshape(-1, 1)
    params, _, _ = initialise(kern)
    kernel_matrix = cross_covariance(kern, x2, x1, params)
    assert kernel_matrix.shape == (n1, n2)
