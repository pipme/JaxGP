# -*- coding: utf-8 -*-
# mypy: ignore-errors
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

import jaxgp as jgp
from jaxgp.contrib.train_utils import train_model
from jaxgp.gpr import GPR
from jaxgp.posteriors import GPRPosterior


@dataclass(frozen=True)
class Datum:
    rng: np.random.RandomState = np.random.RandomState(0)
    X: np.ndarray = rng.randn(100, 1)


def test_GPRPosterior():
    rng = Datum().rng
    X = Datum().X
    # y = np.sin(X @ np.array([[-1.4], [0.5]])) + 0.5 * rng.randn(len(X), 1)
    y = np.sin(X) + 0.01 * rng.randn(len(X), 1)
    train_data = jgp.Dataset(X=X, Y=y)
    kernel = jgp.kernels.RBF(active_dims=(0,))
    model = GPR(train_data=train_data, gprior=jgp.GPrior(kernel=kernel))
    params, constrain_trans, unconstrain_trans = jgp.initialise(model)
    raw_params = unconstrain_trans(params)
    neg_mll = model.build_mll(sign=-1.0)
    print("Initial negative marginal likelihood = ", neg_mll(raw_params))
    solver = jaxopt.ScipyMinimize(
        fun=neg_mll, jit=True, options={"disp": True}
    )
    soln = solver.run(raw_params)
    print(
        "After optimization negative marginal likelihood = ",
        soln.state.fun_val,
    )
    final_params = constrain_trans(soln.params)
    gp_post = model.posterior(final_params)
    X_test = np.linspace(-3, 3, 10)
    pred_mean, pred_var = gp_post.predict_f(X_test)
    assert pred_var.sum() > 0


def test_GPRPosterior_heteroskedastic():
    rng = Datum().rng
    X = Datum().X
    X = np.sort(X, 0)
    noise_variance = jnp.concatenate(
        [
            # 1e-6 * jnp.ones(len(X) // 2),
            jnp.zeros(len(X) // 2),
            jnp.linspace(0.01, 0.2, len(X) - len(X) // 2),
        ]
    )
    y = np.sin(X) + np.sqrt(noise_variance[:, None]) * rng.randn(len(X), 1)
    train_data = jgp.Dataset(X=X, Y=y)
    kernel = jgp.kernels.RBF(active_dims=(0,))
    model = GPR(
        train_data=train_data,
        gprior=jgp.GPrior(kernel=kernel),
        sigma_sq=noise_variance,
    )
    params, constrain_trans, unconstrain_trans = jgp.initialise(model)
    raw_params = unconstrain_trans(params)
    neg_mll = model.build_mll(sign=-1.0)
    print("Initial negative marginal likelihood = ", neg_mll(raw_params))
    solver = jaxopt.ScipyMinimize(
        fun=neg_mll, jit=True, options={"disp": True}
    )
    soln = solver.run(raw_params)
    print(
        "After optimization negative marginal likelihood = ",
        soln.state.fun_val,
    )
    final_params = constrain_trans(soln.params)
    gp_post = model.posterior(final_params)
    X_test = np.linspace(-3, 3, 100)
    pred_mean, pred_var = gp_post.predict_f(X_test)
    # import matplotlib.pyplot as plt
    # plt.scatter(X, y)
    # plt.plot(X_test, pred_mean)
    # plt.show()

    pred_mean, pred_var = gp_post.predict_f(X)
    np.testing.assert_allclose(
        pred_mean[: len(X) // 2], y[: len(X) // 2], atol=1e-4
    )
    assert pred_var.sum() > 0


def test_HeteroskedasticSGPRPosterior_quad_mixture():
    rng = Datum().rng
    D = 2
    X = rng.randn(100, D)
    y = np.sin(np.sum(X**2, 1)) + rng.randn(X.shape[0]) * 0.1

    train_data = jgp.Dataset(X, y)
    mean = jgp.means.Quadratic(input_dim=D)
    kernel = jgp.kernels.RBF(active_dims=tuple(range(D)))
    likelihood = jgp.likelihoods.HeteroskedasticGaussianVBMC(constant_add=True)
    Z = X.copy()
    model = jgp.HeteroskedasticSGPR(
        train_data=train_data,
        gprior=jgp.GPrior(kernel=kernel, mean_function=mean),
        likelihood=likelihood,
        inducing_points=Z,
    )
    _, constrain_trans, unconstrain_trans = jgp.initialise(model)

    # soln = train_model(model)
    # final_params = constrain_trans(soln.params)
    final_params = model.params
    model.post_cache = model.posterior(final_params)

    K = 50
    mu = jnp.array(rng.randn(K, D))
    sigma = jnp.abs(rng.randn(K, D)) + 1e-3
    weights = jnp.ones(K) / K
    import time
    from functools import partial

    start = time.process_time()

    @jax.jit
    @partial(jax.value_and_grad)
    def f(mu, sigma, weights):
        return model.post_cache.quad_mixture(mu, sigma, weights, True, True)[0]

    res = f(mu, sigma, weights)
    print("time: ", time.process_time() - start)
    return
