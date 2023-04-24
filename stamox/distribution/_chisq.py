from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_jit, filter_vmap
from jax._src.random import KeyArray, Shape
from jaxtyping import ArrayLike, Float, Int

from ._gamma import _dgamma, _pgamma, _qgamma
from ._utils import (
    _check_clip_distribution_domain,
    _check_clip_probability,
    _post_process,
    _promote_dtype_to_floating,
)


def dchisq(
    x: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the chi-squared distribution.

    Args:
        x: A float or array-like object representing the values at which to evaluate the chi-squared distribution.
        df: The degrees of freedom for the chi-squared distribution.
        lower_tail: A boolean indicating whether to compute the lower tail of the chi-squared distribution (defaults to True).
        log_prob: A boolean indicating whether to return the log probability (defaults to False).
        dtype: The dtype of the output (defaults to float32).

    Returns:
        ArrayLike: The chi-squared distribution evaluated at `x`.

    Example:
        >>> dchisq(2.0, 3)
        Array([0.20755368], dtype=float32, weak_type=True)
    """
    x, dtype = _promote_dtype_to_floating(x, dtype)
    x = jnp.atleast_1d(x)
    x = _check_clip_distribution_domain(x, 0.0, jnp.inf)
    grads = filter_vmap(_dgamma)(x, df / 2, 1 / 2)
    grads = _post_process(grads, lower_tail, log_prob)
    return grads


def pchisq(
    q: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculates the chi-squared probability density function.

    Args:
        q (Union[float, array-like]): The value of the chi-squared variable.
        df (Union[int, float, array-like]): The degrees of freedom.
        lower_tail (bool): Whether to calculate the lower tail (default True).
        log_prob (bool): Whether to return the log probability (default False).
        dtype (dtype): The dtype of the output (default jnp.float_).

    Returns:
        ArrayLike: The chi-squared probability density function.

    Example:
        >>> pchisq(2.0, 3)
        Array([0.42759317], dtype=float32, weak_type=True)
    """
    q = jnp.asarray(q, dtype=dtype)
    q = jnp.atleast_1d(q)
    q = _check_clip_distribution_domain(q, 0.0, jnp.inf)
    p = filter_vmap(_pgamma)(q, df / 2, 1 / 2)
    p = _post_process(p, lower_tail, log_prob)
    return p


def qchisq(
    p: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the inverse of the chi-squared cumulative distribution function.

    Args:
        p (Union[Float, ArrayLike]): Probability value or array of probability values.
        df (Union[Int, Float, ArrayLike]): Degrees of freedom.
        lower_tail (bool, optional): If True (default), probabilities are P[X ≤ x], otherwise, P[X > x].
        log_prob (bool, optional): If True, probabilities are given as log(p).
        dtype (dtype, optional): The dtype of the output (default jnp.float_).

    Returns:
        ArrayLike: The quantiles corresponding to the given probabilities.

    Example:
        >>> qchisq(0.95, 10)
        Array([18.307034], dtype=float32)
    """
    p = jnp.asarray(p, dtype=dtype)
    p = jnp.atleast_1d(p)
    p = _check_clip_probability(p, lower_tail, log_prob)
    q = filter_vmap(_qgamma)(p, df / 2, 1 / 2)
    return q


@filter_jit
def _rchisq(
    key: KeyArray,
    df: Union[Int, Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
    dtype=jnp.float_,
):
    if sample_shape is None:
        sample_shape = jnp.shape(df)
    df = jnp.broadcast_to(df, sample_shape)
    return jrand.chisquare(key, df, shape=sample_shape, dtype=dtype)


def rchisq(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    df: Union[Int, Float, ArrayLike] = None,
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """
    Generates random variates from the chi-squared distribution.

    Args:
        key (KeyArray): Random key to generate the random numbers.
        sample_shape (Optional[Shape], optional): Shape of the output array. Defaults to None.
        df (Union[Int, Float, ArrayLike], optional): Degrees of freedom. Defaults to None.
        lower_tail (bool, optional): Whether to return the lower tail probability. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.
        dtype (dtype, optional): The dtype of the output (default float_).

    Returns:
        ArrayLike: Random variates from the chi-squared distribution.

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> rchisq(key, df=2)
        Array(1.982825, dtype=float32)
    """
    rvs = _rchisq(key, df, sample_shape, dtype=dtype)
    rvs = _post_process(rvs, lower_tail, log_prob)
    return rvs
