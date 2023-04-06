from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax._src.random import KeyArray, Shape
from jaxtyping import ArrayLike, Float

from ..core import make_partial_pipe


@filter_jit
def _pexp(x: Union[float, ArrayLike], rate: float) -> Float:
    return -jnp.expm1(-rate * x)


@make_partial_pipe
def pexp(
    q: Union[Float, ArrayLike],
    rate: Union[Float, ArrayLike],
    lower_tail: bool = True,
    log_prob: bool = False,
) -> ArrayLike:
    """Calculates the probability of a given value or array of values for an exponential distribution.

    Args:
        q: Union[Float, ArrayLike]. The value or array of values to calculate the probability of.
        rate: Union[Float, ArrayLike]. The rate parameter of the exponential distribution.
        lower_tail: bool, optional. Whether to return the lower tail probability (default is True).
        log_prob: bool, optional. Whether to return the log probability (default is False).

    Returns:
        ArrayLike: The probability of the given value or array of values.

    Example:
        >>> pexp(1.0, 0.5)
        Array([0.39346933], dtype=float32, weak_type=True)
    """
    q = jnp.atleast_1d(q)
    p = filter_vmap(_pexp)(q, rate)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


@filter_jit
def _qexp(q: Union[float, ArrayLike], rate: float) -> Float:
    return -jnp.log1p(-q) / rate


@make_partial_pipe
def qexp(
    p: Union[float, ArrayLike],
    rate: Union[float, ArrayLike],
    lower_tail=True,
    log_prob=False,
) -> ArrayLike:
    """Computes the quantile of an exponential distribution.

    Args:
        p (Union[float, ArrayLike]): Probability or log probability.
        rate (Union[float, ArrayLike]): Rate parameter of the exponential distribution.
        lower_tail (bool, optional): Whether to compute the lower tail. Defaults to True.
        log_prob (bool, optional): Whether `p` is a log probability. Defaults to False.

    Returns:
        ArrayLike: The quantile of the exponential distribution.

    Example:
        >>> qexp(0.5, 1.0)
        Array([0.6931472], dtype=float32, weak_type=True)
    """
    p = jnp.atleast_1d(p)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.exp(p)
    x = filter_vmap(_qexp)(p, rate)
    return x


_dexp = filter_jit(filter_grad(_pexp))


@make_partial_pipe
def dexp(
    x: Union[Float, ArrayLike],
    rate: Union[Float, ArrayLike],
    lower_tail: bool = True,
    log_prob: bool = False,
) -> ArrayLike:
    """Calculates the derivative of the exponential distribution.

    Args:
        x (Union[Float, ArrayLike]): The value at which to evaluate the derivative.
        rate (Union[Float, ArrayLike]): The rate parameter of the exponential distribution.
        lower_tail (bool, optional): Whether to calculate the lower tail probability. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        ArrayLike: The derivative of the exponential distribution evaluated at x.

    Example:
        >>> dexp(1.0, 0.5, lower_tail=True, log_prob=False)
        Array([0.30326533], dtype=float32, weak_type=True)
    """
    x = jnp.atleast_1d(x)
    grads = filter_vmap(_dexp)(x, rate)
    if not lower_tail:
        grads = -grads
    if log_prob:
        grads = jnp.log(grads)
    return grads


@filter_jit
def _rexp(
    key: KeyArray,
    rate: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
):
    if sample_shape is None:
        sample_shape = jnp.shape(rate)
    rate = jnp.broadcast_to(rate, sample_shape)
    return jrand.exponential(key, shape=sample_shape) / rate


@make_partial_pipe
def rexp(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    rate: Union[Float, ArrayLike] = None,
    lower_tail: bool = True,
    log_prob: bool = False,
) -> ArrayLike:
    """Generates random samples from the exponential distribution.

    Args:
        key (KeyArray): A PRNGKey to use for generating random numbers.
        sample_shape (Optional[Shape], optional): The shape of the output array. Defaults to None.
        rate (Union[Float, ArrayLike], optional): The rate parameter of the exponential distribution. Defaults to None.
        lower_tail (bool, optional): Whether to return the lower tail of the distribution. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability of the samples. Defaults to False.

    Returns:
        ArrayLike: An array of random samples from the exponential distribution.

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> rexp(key, sample_shape=(2, 3), rate=1.0, lower_tail=False, log_prob=True)
        Array([[-0.69314718, -0.69314718, -0.69314718],
               [-0.69314718, -0.69314718, -0.69314718]], dtype=float32)
    """
    rvs = _rexp(key, rate, sample_shape)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
