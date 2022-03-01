from jax.config import config

config.update("jax_debug_nans", True)
from dataclasses import dataclass
from operator import ne

import jax.numpy as jnp
from jax import random
from pytest import param
import jaxgp as jgp
import numpy as np
from jaxgp.sgpr import SGPR
import jax
from scipy.optimize import minimize
import jaxopt

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

    train_data = jgp.Dataset(X=X, Y=y)
    kernel = jgp.RBF(active_dims=[0, 1])
    model = SGPR(
        train_data=train_data,
        gprior=jgp.GPrior(kernel=kernel),
        likelihood=jgp.Gaussian(num_datapoints=train_data.N),
        inducing_points=Z,
    )

    params, constrain_trans, unconstrain_trans = jgp.initialise(model)
    raw_params = unconstrain_trans(params)
    neg_elbo = model.build_elbo(sign=-1.0)
    print("Initial negative elbo = ", neg_elbo(raw_params))
    solver = jaxopt.ScipyMinimize(
        fun=neg_elbo, jit=True, options={"disp": True}
    )
    soln = solver.run(raw_params)
    print("After optimization negative elbo = ", soln.state.fun_val)
    # Remember to transform since the optimization is in unconstrained space
    final_params = constrain_trans(soln.params)
    posterior = model.posterior()
    qu_mean, qu_cov = model.compute_qu(final_params)
    f_at_Z_mean, f_at_Z_cov = posterior.predict_f(
        final_params["inducing_points"], final_params, full_cov=True
    )
    assert jnp.allclose(qu_mean, f_at_Z_mean, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(
        qu_cov.reshape(1, 20, 20), f_at_Z_cov, rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
    test_sgpr_qu()
