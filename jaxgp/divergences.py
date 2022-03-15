from typing import Optional

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .config import default_jitter
from .types import Array


def gauss_kl(q_mu: Array, q_sqrt: Array, Kp: Optional[Array] = None):
    """KL divergence KL[q(x) || p(x)] between two multivariate normal dists.

    This function handles sets of independent multivariate normals, e.g.
    independent multivariate normals on each output dimension. It returns
    the sum of the divergences.

    Dists are given by,
          q(x) = N(q_mu, q_sqrt q_sqrt^T)
    and
          p(x) = N(0, Kp)  or  p(x) = N(0, Lp Lp^T)

    :param q_mu: [num_inducing, output_dim]
    :param q_sqrt: [output_dim, num_inducing, num_inducing] or [num_inducing, output_dim]
    :param Kp: [output_dim, num_inducing, num_inducing]
    :return: sum of KL[q(x) || p(x)] for each output_dim with shape []
    """

    if q_sqrt.ndim == 2:
        in_axes = (-1, -1, 0)
    elif q_sqrt.ndim == 3:
        in_axes = (-1, 0, 0)
    else:
        raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))
    if Kp is not None:
        if Kp.ndim == 2:
            Kp = jnp.expand_dims(Kp, 0)
        kls = jax.vmap(single_gauss_kl, in_axes=in_axes)(q_mu, q_sqrt, Kp)
    else:
        kls = jax.vmap(single_gauss_kl, in_axes=in_axes[:-1])(q_mu, q_sqrt)
    kl_sum = jnp.sum(kls)
    return kl_sum


def single_gauss_kl(q_mu: Array, q_sqrt: Array, Kp: Optional[Array] = None):
    """KL divergence KL[q(x) || p(x)] between two multivariate normal dists.

    Dists are given by,
          q(x) = N(q_mu, q_sqrt q_sqrt^T)
    and
          p(x) = N(0, Kp) or p(x) = N(0, I) if Kp is None

    :param q_mu: [num_inducing]
    :param q_sqrt: [num_inducing, num_inducing] or [num_inducing]
    :param Kp: [num_inducing, num_inducing]
    :return: KL[q(x) || p(x)] with shape []
    """
    q_diag = q_sqrt.ndim == 1
    whiten = Kp is None
    num_inducing = q_mu.shape[0]

    if whiten:
        Kp = jnp.eye(num_inducing)
    else:
        Kp = default_jitter(Kp)

    p = tfd.MultivariateNormalFullCovariance(jnp.zeros(q_mu.T.shape), Kp)
    if q_diag:
        q = tfd.MultivariateNormalDiag(q_mu, q_sqrt)
    else:
        q = tfd.MultivariateNormalTriL(q_mu, q_sqrt)
    kl_tf = tfd.kl_divergence(q, p)
    return kl_tf
