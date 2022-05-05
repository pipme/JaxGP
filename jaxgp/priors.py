from typing import Dict

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .helpers import Array


def log_density(param: Array, density: tfd.Distribution) -> Array:
    if density is None:
        log_prob = jnp.array(0.0)
    else:
        log_prob = jnp.sum(density.log_prob(param))
    return log_prob


def recursive_items(d1: Dict, d2: Dict):
    for key, value in d1.items():
        if type(value) is dict:
            yield from recursive_items(value, d2[key])
        else:
            yield (key, value, d2[key])


def evaluate_priors(params: Dict, priors: Dict) -> Dict:
    """Recursive loop over pair of dictionaries that correspond to a parameter's
    current value and the parameter's respective prior distribution. For
    parameters where a prior distribution is specified, the log-prior density is
    evaluated at the parameter's current value.

    Args: params (dict): Dictionary containing the current set of parameter
        estimates. priors (dict): Dictionary specifying the parameters' prior
        distributions.

    Returns: Array: The log-prior density, summed over all parameters.
    """
    lpd = jnp.array(0.0)
    if priors is not None:
        for name, param, prior in recursive_items(params, priors):
            lpd += log_density(param, prior)
    return lpd
