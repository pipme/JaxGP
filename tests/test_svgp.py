# -*- coding: utf-8 -*-
# mypy: ignore-errors

from jax.config import config

config.update("jax_debug_nans", True)

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

import jaxgp as jgp
from jaxgp.sgpr import SGPR


def test_svgp_predict():
    input_dim = 1
    output_dim = 2
    num_data = 100
    num_test = 100
    num_inducing = 50
    batch_size = 60

    def func(X):
        return np.sin(2 * X) + 0.3 * X + np.random.normal(0, 0.1, X.shape)

    X = np.random.uniform(-3.0, 3.0, (num_data, input_dim))
    Y = func(X)

    key = jax.random.PRNGKey(10)

    Xtest = jnp.sort(
        jax.random.uniform(
            key, shape=(num_test, input_dim), minval=-5, maxval=5
        ),
        0,
    )

    mean = jgp.means.Quadratic()
    kernel = jgp.kernels.RBF()
    gprior = jgp.GPrior(kernel=kernel, mean_function=mean)
    likelihood = jgp.likelihoods.Gaussian()
    inducing_points = (
        jax.random.uniform(key=key, shape=(num_inducing, input_dim))
        * (X.max() - X.min())
        + X.min()
    )
    model = jgp.SVGP(gprior, likelihood, inducing_points, output_dim)
    params, constrain_trans, unconstrain_trans = jgp.initialise(model)
    raw_params = unconstrain_trans(params)
    neg_elbo = model.build_elbo(num_data=num_data, sign=-1.0)
    batch = jgp.Dataset(X, Y)
    neg_elbo(raw_params, batch)
    model.predict_y(params, Xtest, True)
