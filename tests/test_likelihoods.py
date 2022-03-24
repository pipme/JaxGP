# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from jaxgp.likelihoods import Gaussian


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
