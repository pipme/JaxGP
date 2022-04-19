# -*- coding: utf-8 -*-
# mypy: ignore-errors

from jax.config import config

config.update("jax_debug_nans", True)
from dataclasses import dataclass

import jax.numpy as jnp
import jaxopt
import numpy as np

import jaxgp as jgp
from jaxgp.sgpr_heteroskedastic import HeteroskedasticSGPR


@dataclass(frozen=True)
class Datum:
    rng: np.random.RandomState = np.random.RandomState(0)
    X: np.ndarray = rng.randn(100, 2)
    Z: np.ndarray = rng.randn(20, 2)


def test_heteroskedastic_sgpr_qu():
    rng = Datum().rng
    X = Datum().X
    Z = Datum().Z
    y = np.sin(X @ np.array([[-1.4], [0.5]])) + 0.5 * rng.randn(len(X), 1)

    train_data = jgp.Dataset(X=X, Y=y)
    kernel = jgp.kernels.RBF(active_dims=[0, 1])
    model = HeteroskedasticSGPR(
        train_data=train_data,
        gprior=jgp.GPrior(kernel=kernel),
        likelihood=jgp.likelihoods.FixedHeteroskedasticGaussian(),
        sigma_sq_user=jnp.ones(X.shape[0]),
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
