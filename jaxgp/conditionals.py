from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg

from .config import default_jitter
from .helpers import Array
from .kernels import Kernel, cross_covariance, gram


def conditional(
    kernel_params: Dict,
    Xnew: Array,
    X: Array,
    kernel: Kernel,
    f: Array,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[Array] = None,
    whiten: bool = False,
) -> Tuple[Array, Array]:
    """GP Conditional.

    Multidispatch handles changing implementation for multioutput etc
    Xnew: [N, D]
    X: [M, D]
    f: [M, output_dim] pr [M,]
    q_sqrt: [output_dim, M, M] or [M, M] or [M, output_dim] or [M]
    """
    f_mean, f_cov = single_output_conditional(
        kernel_params,
        Xnew,
        # inducing_variable,
        X,
        kernel,
        f=f,
        full_cov=full_cov,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    return f_mean, f_cov


def single_output_conditional(
    kernel_params: Dict,
    Xnew: Array,
    X: Array,
    # inducing_variable: InducingVariable,
    kernel: Kernel,
    f: Array,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[Array] = None,
    whiten: bool = False,
) -> Tuple[Array, Array]:
    """Single-output GP conditional.
    Xnew: [N, D]
    X: [M, D]
    f: [M, P]
    q_sqrt: [P, M, M] or [M, P]
    """
    Kmm = gram(kernel, X, kernel_params)
    Kmm = default_jitter(Kmm)
    Kmn = cross_covariance(kernel, X, Xnew, kernel_params)  # [M, N]
    Knn = gram(kernel, Xnew, kernel_params, full_cov=full_cov)  # [N, N]

    # setup axis containing output dim which are to be mapped over
    if full_cov:  # [output_dim, num_data, num_data]
        out_axes = (-1, 0)
    else:  # [num_data, output_dim]
        out_axes = (-1, -1)

    def base_conditional_wrapper(f_, q_sqrt_):  # type: ignore
        return base_conditional(
            Kmn,
            Kmm,
            Knn,
            f_,
            full_cov=full_cov,
            q_sqrt=q_sqrt_,
            whiten=whiten,
        )

    if q_sqrt is not None:
        if q_sqrt.ndim == 2:  # [M, num_latent_gps]
            in_axes = (-1, -1)
        elif q_sqrt.ndim == 3:  # [num_latent_gps, M, M]
            in_axes = (-1, 0)
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

        f_mean, f_cov = jax.vmap(
            base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes
        )(f, q_sqrt)
    else:
        f_mean, f_cov = jax.vmap(
            base_conditional_wrapper, in_axes=-1, out_axes=out_axes
        )(f, q_sqrt)
    return f_mean, f_cov


def base_conditional(
    Kmn: Array,
    Kmm: Array,
    Knn: Array,
    f: Array,
    full_cov: Optional[bool] = False,
    q_sqrt: Optional[Array] = None,
    whiten: Optional[bool] = False,
) -> Tuple[Array, Array]:
    r"""Base conditional for single outputs.

    Handling of output dimensions (independent/correlated) will be separate.

    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2; 0, Kmm)
      p(g1) = N(g1; 0, Knn)
      p(g1 | g2) = N(g1; Knm (Kmm⁻¹) g2, Knn - Knm (Kmm⁻¹) Kmn)
    And
      q(g2) = N(g2; f, q_sqrt q_sqrtᵀ)
    This method computes the mean and (co)variance of
      q(g1) = ∫ q(g2) p(g1 | g2) =

    If not whiten: q(g1) = N(g1; Knm Kmm^{-1} f, Knm Kmm^{-1} q_sqrt q_sqrt^T Kmm^{-1} Kmn + Knn - Knm (Kmm⁻¹) Km)
    If whiten: q(g1) = N(g1; Knm Lm^{T}^{-1} f, Knm Lm^{T}^{-1} q_sqrt q_sqrt^T Lm^{-1} Kmn + Knn - Knm (Kmm⁻¹) Km)

    :param Kmn: [M, N], M = number of inducing points
    :param Kmm: [M, M]
    :param Knn: [N, N]  or  [N]
    :param f: [M]
    :param full_cov: bool
    :param q_sqrt: [M, M] (lower triangular) or [M] (diagonal)
    :param whiten: bool
    :return: mean [N] and (co)variance [N]  or [N, N]
    """
    Lm = linalg.cholesky(Kmm, lower=True)
    return base_conditional_with_lm(
        Kmn=Kmn,
        Lm=Lm,
        Knn=Knn,
        f=f,
        full_cov=full_cov,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )


def base_conditional_with_lm(
    Kmn: Array,
    Lm: Array,
    Knn: Array,
    f: Array,
    full_cov: Optional[bool] = False,
    q_sqrt: Optional[Array] = None,
    whiten: Optional[bool] = False,
) -> Tuple[Array, Array]:
    """Same as base_conditional but expects the cholesky Lm instead of Kmm = Lm Lm.T

    Lm can be precomputed, improving performance.
    """
    A = linalg.solve_triangular(Lm, Kmn, lower=True)  # [M, N]

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - jnp.matmul(A.T, A)
    else:
        if Knn.ndim == 2:
            Knn = Knn.diagonal()
        fvar = Knn - jnp.sum(jnp.square(A), 0)

    # another backsubstitution in the unwhitened case
    if not whiten:
        A = linalg.solve_triangular(Lm.T, A, lower=False)  # type: ignore #  [M, N]

    # conditional mean
    fmean = A.T @ f  # [N]

    # covariance due to inducing variables
    if q_sqrt is not None:
        if q_sqrt.ndim == 1:  # diagonal, [M,]
            LTA = jnp.expand_dims(q_sqrt, axis=-1) * A  # [M, N]
        elif q_sqrt.ndim == 2:
            LTA = q_sqrt.T @ A  # type: ignore # [M, N]
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

        if full_cov:
            fvar = fvar + LTA.T @ LTA  # [N, N]
        else:
            fvar = fvar + jnp.sum(jnp.square(LTA), 0)  # [N]

    return fmean, fvar
