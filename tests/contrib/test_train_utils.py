# -*- coding: utf-8 -*-
# mypy: ignore-errors
from dataclasses import dataclass

import numpy as np

import jaxgp as jgp
from jaxgp.contrib.train_utils import train_model_separate


@dataclass(frozen=True)
class Datum:
    rng: np.random.RandomState = np.random.RandomState(0)


def test_train_model_seperate():
    rng = Datum().rng
    D = 2
    X = rng.randn(50, D)
    y = np.sin(np.sum(X**2, 1)) + rng.randn(X.shape[0]) * 0.1

    train_data = jgp.Dataset(X, y)
    mean = jgp.means.Quadratic(input_dim=D)
    kernel = jgp.kernels.RBF(active_dims=tuple(range(D)))
    likelihood = jgp.likelihoods.HeteroskedasticGaussianVBMC(constant_add=True)
    Z = X[:5].copy()
    model = jgp.HeteroskedasticSGPR(
        train_data=train_data,
        gprior=jgp.GPrior(kernel=kernel, mean_function=mean),
        likelihood=likelihood,
        inducing_points=Z,
    )
    params, constrain_trans, unconstrain_trans = jgp.initialise(model)

    inds = np.array([True, True, False, False, False])
    diff_params = {"inducing_points": inds}
    soln = train_model_separate(model, diff_params, options={"disp": True})
    final_params = constrain_trans(soln.params)
    assert final_params["inducing_points"].shape == (np.sum(inds), 2)
