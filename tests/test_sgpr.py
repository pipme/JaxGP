from dataclasses import dataclass
import imp

import jax.numpy as jnp
from jax import random
import jaxgp as jgp
import numpy as np
from jaxgp.sgpr import SGPR

# @dataclass(frozen=True)
# class Datum:
#     key: random.KeyArray = random.PRNGKey(42)
#     X: Array = random.normal(key, (100, 2))
#     Y: Array = random.normal(key, (100, 1))
#     Z: Array = random.normal(key, (10, 2))
#     Xs: Array = random.normal(key, (10, 1))
#     lik = gpflow.likelihoods.Gaussian()
#     kernel = gpflow.kernels.Matern32()
@dataclass(frozen=True)
class Datum:
    rng: np.random.RandomState = np.random.RandomState(0)
    X: np.ndarray = rng.randn(100, 2)
    Z: np.ndarray = rng.randn(20, 2)


def test_sgpr_qu():
    rng = Datum().rng
    X = Datum().X
    Z = Datum().Z
    y = np.sin(X @ np.array([[-1.4], [0.5]])) + 0.5 * rng.randn(len(X), 1)

    train_data = jgp.Dataset(X=X, y=y)
    kernel = jgp.RBF(active_dims=[0, 1])
    sgpr = SGPR(
        train_data=train_data,
        gprior=jgp.GPrior(kernel),
        likelihood=jgp.Gaussian(num_datapoints=train_data.N),
    )

    jgp.initialise()
    # model = gpflow.models.SGPR(
    #     (X, Y), kernel=gpflow.kernels.SquaredExponential(), inducing_variable=Z
    # )

    # gpflow.optimizers.Scipy().minimize(
    #     model.training_loss, variables=model.trainable_variables
    # )

    # qu_mean, qu_cov = model.compute_qu()
    # f_at_Z_mean, f_at_Z_cov = model.predict_f(
    #     model.inducing_variable.Z, full_cov=True
    # )

    # np.testing.assert_allclose(qu_mean, f_at_Z_mean, rtol=1e-5, atol=1e-5)
    # np.testing.assert_allclose(
    #     tf.reshape(qu_cov, (1, 20, 20)), f_at_Z_cov, rtol=1e-5, atol=1e-5
    # )
