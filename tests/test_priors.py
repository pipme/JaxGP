# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from jaxgp.priors import TruncatedPrior, evaluate_priors, log_density
from jaxgp.utils import copy_dict_structure


def test_none_prior():
    """
    Test that multiple dispatch is working in the case of no priors.
    """
    params = {
        "kernel": {
            "lengthscale": jnp.array([1.0]),
            "variance": jnp.array([1.0]),
        },
        "likelihood": {"noise": jnp.array([1.0])},
    }
    priors = copy_dict_structure(params)
    lpd = evaluate_priors(params, priors)
    assert lpd == 0.0


def test_multi_dimensional_priors():
    params = {
        "kernel": {
            "lengthscale": jnp.array([1.0, 2.0]),
        },
        "likelihood": {"noise": jnp.array([1.0])},
    }
    priors = copy_dict_structure(params)
    concentrations = [1.0, 2.0]
    rates = [1.0, 3.0]
    priors["kernel"]["lengthscale"] = tfd.Gamma(concentrations, rates)

    lpd_1 = evaluate_priors(params, priors)
    lpd_2 = 0.0
    for i in range(len(concentrations)):
        lpd_2 += tfd.Gamma(concentrations[i], rates[i]).log_prob(
            params["kernel"]["lengthscale"][i]
        )
    assert np.isclose(lpd_1, lpd_2)


def test_TruncatedPrior():
    prior = TruncatedPrior(tfd.Normal(0, 1), lower_bounds=-1, upper_bounds=2)
    p1 = prior.log_prob(0.0)
    p2 = tfd.TruncatedNormal(0, 1, -1, 2).log_prob(0.0)
    assert p1 == p2

    prior = TruncatedPrior(
        tfd.Normal([0, 1], [1, 2]), lower_bounds=[-1, -2], upper_bounds=[2, 3]
    )
    p1 = prior.log_prob([0.0, 1.0])
    p2 = jnp.array(
        [
            tfd.TruncatedNormal(0, 1, -1, 2).log_prob(0.0),
            tfd.TruncatedNormal(1, 2, -2, 3).log_prob(1.0),
        ]
    )
    assert np.allclose(p1, p2)
