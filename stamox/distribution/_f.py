from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax._src.random import KeyArray, Shape
from jaxtyping import ArrayLike, Float

from ..math.special import fdtr, fdtri
from ._utils import (
    _check_clip_distribution_domain,
    _check_clip_probability,
    _post_process,
    _promote_dtype_to_floating,
)


@filter_jit
def _pf(
    x: Union[Float, ArrayLike],
    dfn: Union[Float, ArrayLike],
    dfd: Union[Float, ArrayLike],
):
    dfn = jnp.asarray(dfn, dtype=x.dtype)
    dfd = jnp.asarray(dfd, dtype=x.dtype)
    return fdtr(dfn, dfd, x)


def pF(
    q: Union[Float, ArrayLike],
    dfn: Union[Float, ArrayLike],
    dfd: Union[Float, ArrayLike],
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculates the cumulative distribution function of the F-distribution.

    Args:
        q (Union[Float, ArrayLike]): The value at which to evaluate the cdf.
        dfn (Union[Float, ArrayLike]): The numerator degrees of freedom.
        dfd (Union[Float, ArrayLike]): The denominator degrees of freedom.
        lower_tail (bool, optional): If True (default), the lower tail probability is returned.
        log_prob (bool, optional): If True, the logarithm of the probability is returned.
        dtype (jnp.dtype, optional): The dtype of the output (default is jnp.float_).

    Returns:
        ArrayLike: The cumulative distribution function evaluated at `q`.

    Example:
        >>> pF(1.0, 1.0, 1.0)
        Array([0.5000001], dtype=float32, weak_type=True)
    """
    q, _ = _promote_dtype_to_floating(q, dtype)
    q = jnp.atleast_1d(q)
    q = _check_clip_distribution_domain(q, 0)
    p = filter_vmap(_pf)(q, dfn, dfd)
    p = _post_process(p, lower_tail, log_prob)
    return p


_df = filter_jit(filter_grad(_pf))


def dF(
    x: Union[Float, ArrayLike],
    dfn: Union[Float, ArrayLike],
    dfd: Union[Float, ArrayLike],
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculates the gradient of the cumulative distribution function for a given x, dfn and dfd.

    Args:
        x (Union[Float, ArrayLike]): The value at which to calculate the gradient of the cumulative distribution function.
        dfn (Union[Float, ArrayLike]): The numerator degrees of freedom.
        dfd (Union[Float, ArrayLike]): The denominator degrees of freedom.
        lower_tail (bool, optional): Whether to calculate the lower tail of the cumulative distribution function. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The gradient of the cumulative distribution function.

    Example:
        >>> dF(1.0, 1.0, 1.0)
        Array([0.1591549], dtype=float32, weak_type=True)
    """
    x, _ = _promote_dtype_to_floating(x, dtype)
    x = jnp.atleast_1d(x)
    x = _check_clip_distribution_domain(x, 0)
    grads = filter_vmap(_df)(x, dfn, dfd)
    grads = _post_process(grads, lower_tail, log_prob)
    return grads


@filter_jit
def _qf(
    q: Union[Float, ArrayLike],
    dfn: Union[Float, ArrayLike],
    dfd: Union[Float, ArrayLike],
):
    dfn = jnp.asarray(dfn, dtype=q.dtype)
    dfd = jnp.asarray(dfd, dtype=q.dtype)
    return fdtri(dfn, dfd, q)


def qF(
    p: Union[Float, ArrayLike],
    dfn: Union[Float, ArrayLike],
    dfd: Union[Float, ArrayLike],
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculates the quantile function of a given distribution.

    Args:
        p (Union[Float, ArrayLike]): The quantile to calculate.
        dfn (Union[Float, ArrayLike]): The degrees of freedom for the numerator.
        dfd (Union[Float, ArrayLike]): The degrees of freedom for the denominator.
        lower_tail (bool, optional): Whether to calculate the lower tail or not. Defaults to True.
        log_prob (bool, optional): Whether to calculate the log probability or not. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The calculated quantile.

    Example:
        >>> qF(0.5, 1.0, 1.0)
        Array([0.99999714], dtype=float32)
    """
    p, _ = _promote_dtype_to_floating(p, dtype)
    p = jnp.atleast_1d(p)
    p = _check_clip_probability(p, lower_tail, log_prob)
    return filter_vmap(_qf)(p, dfn, dfd)


@filter_jit
def _rf(
    key: KeyArray,
    dfn: Union[Float, ArrayLike],
    dfd: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
    dtype=jnp.float_,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(dfn), jnp.shape(dfd))
    dfn = jnp.broadcast_to(dfn, sample_shape)
    dfd = jnp.broadcast_to(dfd, sample_shape)
    return jrand.f(key, dfn, dfd, shape=sample_shape, dtype=dtype)


def rF(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    dfn: Union[Float, ArrayLike] = None,
    dfd: Union[Float, ArrayLike] = None,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
):
    """Generate random variates from F-distribution.

    Args:
        key (KeyArray): Random key used for PRNG.
        sample_shape (Optional[Shape], optional): Shape of the samples to be drawn. Defaults to None.
        dfn (Union[Float, ArrayLike]): Degrees of freedom in numerator.
        dfd (Union[Float, ArrayLike]): Degrees of freedom in denominator.
        lower_tail (bool, optional): Whether to calculate the lower tail probability. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to float32.

    Returns:
        ArrayLike : Random variates from F-distribution.

    Example:
        >>> rF(jax.random.PRNGKey(0), dfn=1.0, dfd=1.0)
        Array(40.787617, dtype=float32)

    """
    rvs = _rf(key, dfn, dfd, sample_shape, dtype=dtype)
    rvs = _post_process(rvs, lower_tail, log_prob)
    return rvs
