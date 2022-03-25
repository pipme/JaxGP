# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from jaxgp.likelihoods import Gaussian, HeteroskedasticGaussian


def test_gaussian_predict_mean_and_var_shape():
    likelihood = Gaussian()
    params = likelihood.params
    N = 3
    Fmu = jnp.zeros(N)
    Fvar = jr.normal(
        jr.PRNGKey(42), (2, N, N)
    )  # not really a valid covariance matrix, just for test
    _, Yvar = likelihood.predict_mean_and_var(params, Fmu, Fvar, True)
    Yvar_np = np.array(Fvar)
    for i in range(Fvar.shape[0]):
        np.fill_diagonal(Yvar_np[i], Fvar[i].diagonal() + params["noise"])
    np.testing.assert_allclose(Yvar, Yvar_np)


def test_heteroskedastic_gaussian():
    likelihood = Gaussian()
    params = likelihood.params
    N = 3
    Fmu = jnp.zeros(N)
    Fvar = jr.normal(
        jr.PRNGKey(42), (2, N, N)
    )  # not really a valid covariance matrix, just for test
    _, Yvar = likelihood.predict_mean_and_var(params, Fmu, Fvar, True)
    Yvar_np = np.array(Fvar)
    for i in range(Fvar.shape[0]):
        np.fill_diagonal(Yvar_np[i], Fvar[i].diagonal() + params["noise"])
    np.testing.assert_allclose(Yvar, Yvar_np)


def test_heteroskedastic_gaussian_predict_mean_and_var_shape():
    likelihood = HeteroskedasticGaussian(user_provided=True)
    params = likelihood.params
    keys = jr.split(jr.PRNGKey(42))
    N = 3
    sigma_sq = jnp.abs(jr.normal(keys[0], (N,)))
    Fmu = jnp.zeros(N)
    Fvar = jr.normal(
        keys[1], (2, N, N)
    )  # not really a valid covariance matrix, just for test
    _, Yvar = likelihood.predict_mean_and_var(
        params, Fmu, Fvar, True, sigma_sq
    )
    Yvar_np = np.array(Fvar)
    for i in range(Fvar.shape[0]):
        np.fill_diagonal(Yvar_np[i], Fvar[i].diagonal() + sigma_sq)
    np.testing.assert_allclose(Yvar, Yvar_np)
