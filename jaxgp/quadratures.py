from typing import Callable
import numpy as np
import jax.numpy as jnp

"""The number of Gauss-Hermite points to use for quadrature"""
DEFAULT_NUM_GAUSS_HERMITE_POINTS = 20

def gh_points_and_weights(n_gh: int):
    r"""
    Given the number of Gauss-Hermite points n_gh,
    returns the points z and the weights dz to perform the following
    uni-dimensional gaussian quadrature:

    X ~ N(mean, stddev²)
    E[f(X)] = ∫ f(x) p(x) dx = \sum_{i=1}^{n_gh} f(mean + stddev*z_i) dz_i

    :param n_gh: Number of Gauss-Hermite points
    :returns: Points z and weights dz, both tensors with shape [n_gh],
        to compute uni-dimensional gaussian expectation
    """
    z, dz = np.polynomial.hermite.hermgauss(n_gh)
    z = z * np.sqrt(2)
    dz = dz / np.sqrt(np.pi)
    return z, dz

def gauss_hermite_quadrature(
    fun: Callable,
    mean,
    var,
    deg: int = DEFAULT_NUM_GAUSS_HERMITE_POINTS,
    *args,
    **kwargs
):
    gh_points, gh_weights = gh_points_and_weights(deg)
    stdev = jnp.sqrt(var)
    X = mean + stdev * gh_points
    W = gh_weights
    return jnp.sum(fun(X, *args, **kwargs) * W, axis=0)
