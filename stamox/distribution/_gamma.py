from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
import tensorflow_probability.substrates.jax.math as tfp_math
from equinox import filter_grad, filter_jit
from jax import lax
from jax._src.random import KeyArray, Shape
from jax.scipy.special import gammainc
from jaxtyping import ArrayLike, Float

from ._utils import (
    _check_clip_distribution_domain,
    _check_clip_probability,
    _post_process,
    _promote_dtype_to_floating,
    svmap_,
)


@filter_jit
def _pgamma(
    q, shape: Union[Float, ArrayLike] = 1.0, rate: Union[Float, ArrayLike] = 1.0
):
    return gammainc(shape, q * rate)


def pgamma(
    q: Union[Float, ArrayLike],
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the cumulative distribution function of the gamma distribution.

    Args:
        q: A float or array-like object representing the input to the gamma function.
        shape: A float or array-like object representing the shape parameter of the gamma function.
        rate: A float or array-like object representing the rate parameter of the gamma function.
        lower_tail: A boolean indicating whether to compute the lower tail of the gamma function.
        log_prob: A boolean indicating whether to compute the logarithm of the probability density function.
        dtype: The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The CDF value of the given value or array of values.

    Example:
        >>> pgamma(1.0, 0.5, 0.5)
        Array(0.6826893, dtype=float32, weak_type=True)
    """
    q, _ = _promote_dtype_to_floating(q, dtype)
    q = _check_clip_distribution_domain(q, 0.0, jnp.inf)
    p = svmap_(_pgamma, q, shape, rate)
    p = _post_process(p, lower_tail, log_prob)
    return p


_dgamma = filter_jit(filter_grad(_pgamma))


def dgamma(
    x: Union[Float, ArrayLike],
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Compute density of gamma distribution.

    Args:
        x (Union[Float, ArrayLike]): The value at which to evaluate the gamma
            distribution.
        shape (Union[Float, ArrayLike], optional): The shape parameter of the
            gamma distribution. Defaults to 1.0.
        rate (Union[Float, ArrayLike], optional): The rate parameter of the
            gamma distribution. Defaults to 1.0.
        lower_tail (bool, optional): Whether to compute the lower tail of the
            gamma distribution. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability.
            Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to

    Returns:
        ArrayLike: The density of the gamma distribution evaluated
            at `x`. If `log_prob` is True, returns the log probability.

    Example:
        >>> dgamma(1.0, 0.5, 0.5)
        Array(0.24197064, dtype=float32, weak_type=True)
    """
    x, _ = _promote_dtype_to_floating(x, dtype)
    x = _check_clip_distribution_domain(x, 0.0, jnp.inf)
    grads = svmap_(_dgamma, x, shape, rate)
    grads = _post_process(grads, lower_tail, log_prob)
    return grads


@filter_jit
def _qgamma(
    q: Union[Float, ArrayLike],
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
):
    dtype = lax.dtype(q)
    shape = jnp.asarray(shape, dtype=dtype)
    rate = jnp.asarray(rate, dtype=dtype)
    return lax.div(tfp_math.igammainv(shape, q), rate)


def qgamma(
    p: Union[Float, ArrayLike],
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the quantile of the gamma distribution.

    Args:
        p: A float or array-like object representing the quantile.
        shape: A float or array-like object representing the shape parameter of the gamma distribution.
        rate: A float or array-like object representing the rate parameter of the gamma distribution.
        lower_tail: A boolean indicating whether to compute the lower tail (default) or upper tail.
        log_prob: A boolean indicating whether to compute the log probability (default False).
        dtype: The dtype of the output. Defaults to float_.

    Returns:
        ArrayLike: The quantile of the gamma distribution.

    Example:
        >>> qgamma(0.5, 0.5, 0.5)
        Array([0.45493677], dtype=float32)
    """
    p, _ = _promote_dtype_to_floating(p, dtype)
    p = _check_clip_probability(p, lower_tail, log_prob)
    x = svmap_(_qgamma, p, shape, rate)
    return x


def rgamma(
    key,
    sample_shape: Optional[Shape] = None,
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Generates random gamma values.

    Args:
        key: A PRNGKey to use for the random number generation.
        sample_shape: An optional shape for the output array.
        shape: The shape parameter of the gamma distribution.
        rate: The rate parameter of the gamma distribution.
        lower_tail: Whether to return the lower tail of the distribution.
        log_prob: Whether to return the log probability of the result.
        dtype: The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: A random gamma value or an array of random gamma values.

    Example:
        >>> rgamma(key, shape=0.5, rate=0.5)
        Array(0.3384059, dtype=float32)
    """
    rvs = _rgamma(key, shape, rate, sample_shape, dtype=dtype)
    rvs = _post_process(rvs, lower_tail, log_prob)
    return rvs


@filter_jit
def _rgamma(
    key: KeyArray,
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
    dtype=jnp.float_,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(shape), jnp.shape(rate))
    shape = jnp.broadcast_to(shape, sample_shape)
    rate = jnp.broadcast_to(rate, sample_shape)
    return jrand.gamma(key, shape, sample_shape, dtype) / rate
