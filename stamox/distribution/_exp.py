from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax import lax
from jax._src.random import KeyArray, Shape
from jaxtyping import ArrayLike, Float

from ._utils import (
    _check_clip_distribution_domain,
    _check_clip_probability,
    _post_process,
    _promote_dtype_to_floating,
)


@filter_jit
def _pexp(x: Union[float, ArrayLike], rate: float) -> Float:
    return -jnp.expm1(-rate * x)


def pexp(
    q: Union[Float, ArrayLike],
    rate: Union[Float, ArrayLike],
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculates the probability of a given value or array of values for an exponential distribution.

    Args:
        q: Union[Float, ArrayLike]. The value or array of values to calculate the probability of.
        rate: Union[Float, ArrayLike]. The rate parameter of the exponential distribution.
        lower_tail: bool, optional. Whether to return the lower tail probability (default is True).
        log_prob: bool, optional. Whether to return the log probability (default is False).
        dtype: jnp.dtype, optional. The dtype of the output (default is float_).

    Returns:
        ArrayLike: The probability of the given value or array of values.

    Example:
        >>> pexp(1.0, 0.5)
        Array([0.39346933], dtype=float32, weak_type=True)
    """
    q, _ = _promote_dtype_to_floating(q, dtype)
    q = jnp.atleast_1d(q)
    q = _check_clip_distribution_domain(q, 0)
    p = filter_vmap(_pexp)(q, rate)
    p = _post_process(p, lower_tail, log_prob)
    return p


@filter_jit
def _qexp(q: Union[float, ArrayLike], rate: float) -> Float:
    dtype = lax.dtype(q)
    rate = lax.convert_element_type(rate, dtype)
    return -lax.div(lax.log1p(-q), rate)


def qexp(
    p: Union[float, ArrayLike],
    rate: Union[float, ArrayLike],
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the quantile of an exponential distribution.

    Args:
        p (Union[float, ArrayLike]): Probability or log probability.
        rate (Union[float, ArrayLike]): Rate parameter of the exponential distribution.
        lower_tail (bool, optional): Whether to compute the lower tail. Defaults to True.
        log_prob (bool, optional): Whether `p` is a log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The quantile of the exponential distribution.

    Example:
        >>> qexp(0.5, 1.0)
        Array([0.6931472], dtype=float32, weak_type=True)
    """
    p, _ = _promote_dtype_to_floating(p, dtype)
    p = jnp.atleast_1d(p)
    p = _check_clip_probability(p, lower_tail, log_prob)
    x = filter_vmap(_qexp)(p, rate)
    return x


_dexp = filter_jit(filter_grad(_pexp))


def dexp(
    x: Union[Float, ArrayLike],
    rate: Union[Float, ArrayLike],
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculates the derivative of the exponential distribution.

    Args:
        x (Union[Float, ArrayLike]): The value at which to evaluate the derivative.
        rate (Union[Float, ArrayLike]): The rate parameter of the exponential distribution.
        lower_tail (bool, optional): Whether to calculate the lower tail probability. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The derivative of the exponential distribution evaluated at x.

    Example:
        >>> dexp(1.0, 0.5, lower_tail=True, log_prob=False)
        Array([0.30326533], dtype=float32, weak_type=True)
    """
    x, _ = _promote_dtype_to_floating(x, dtype)
    x = jnp.atleast_1d(x)
    x = _check_clip_distribution_domain(x, 0)
    grads = filter_vmap(_dexp)(x, rate)
    grads = _post_process(grads, lower_tail, log_prob)
    return grads


@filter_jit
def _rexp(
    key: KeyArray,
    rate: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
    dtype=jnp.float_,
):
    if sample_shape is None:
        sample_shape = jnp.shape(rate)
    rate = jnp.broadcast_to(rate, sample_shape)
    return jrand.exponential(key, shape=sample_shape, dtype=dtype) / rate


def rexp(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    rate: Union[Float, ArrayLike] = None,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Generates random samples from the exponential distribution.

    Args:
        key (KeyArray): A PRNGKey to use for generating random numbers.
        sample_shape (Optional[Shape], optional): The shape of the output array. Defaults to None.
        rate (Union[Float, ArrayLike], optional): The rate parameter of the exponential distribution. Defaults to None.
        lower_tail (bool, optional): Whether to return the lower tail of the distribution. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability of the samples. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: An array of random samples from the exponential distribution.

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> rexp(key, sample_shape=(2, 3), rate=1.0, lower_tail=False, log_prob=True)
        Array([[-0.69314718, -0.69314718, -0.69314718],
               [-0.69314718, -0.69314718, -0.69314718]], dtype=float32)
    """
    rvs = _rexp(key, rate, sample_shape, dtype)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
