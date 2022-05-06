from typing import Dict

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .helpers import Array, dataclass


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


@dataclass(frozen=True)
class TruncatedPrior:
    """A simple wrapper around tfd.Distribution for evaluating truncated prior's log density."""

    base_prior: tfd.Distribution
    lower_bounds: Array
    upper_bounds: Array

    def __post_init__(self) -> None:
        normalising_constant = self.base_prior.cdf(
            self.upper_bounds
        ) - self.base_prior.cdf(self.lower_bounds)
        object.__setattr__(self, "normalising_constant", normalising_constant)
        object.__setattr__(
            self, "log_normalising_constant", jnp.log(normalising_constant)
        )

    def log_prob(self, x: Array) -> Array:
        return self.base_prior.log_prob(x) - self.log_normalising_constant  # type: ignore
