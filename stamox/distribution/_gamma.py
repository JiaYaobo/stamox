from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
import tensorflow_probability.substrates.jax.math as tfp_math
from equinox import filter_grad, filter_jit, filter_vmap
from jax import lax
from jax._src.random import KeyArray, Shape
from jax.scipy.special import gammainc
from jaxtyping import ArrayLike, Float


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
    dtype=jnp.float32,
) -> ArrayLike:
    """Computes the cumulative distribution function of the gamma distribution.

    Args:
        q: A float or array-like object representing the input to the gamma function.
        shape: A float or array-like object representing the shape parameter of the gamma function.
        rate: A float or array-like object representing the rate parameter of the gamma function.
        lower_tail: A boolean indicating whether to compute the lower tail of the gamma function.
        log_prob: A boolean indicating whether to compute the logarithm of the probability density function.
        dtype: The dtype of the output. Defaults to float32.

    Returns:
        ArrayLike: The CDF value of the given value or array of values.

    Example:
        >>> pgamma(1.0, 0.5, 0.5)
        Array([0.6826893], dtype=float32, weak_type=True)
    """
    q = jnp.asarray(q, dtype=dtype)
    q = jnp.atleast_1d(q)
    p = filter_vmap(_pgamma)(q, shape, rate)
    if not lower_tail:
        p = 1.0 - p
    if log_prob:
        p = jnp.log(p)
    return p


_dgamma = filter_jit(filter_grad(_pgamma))


def dgamma(
    x: Union[Float, ArrayLike],
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float32,
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
        Array([0.24197064], dtype=float32, weak_type=True)
    """
    x = jnp.asarray(x, dtype=dtype)
    x = jnp.atleast_1d(x)
    grads = filter_vmap(_dgamma)(x, shape, rate)
    if not lower_tail:
        grads = 1 - grads
    if log_prob:
        grads = jnp.log(grads)
    return grads


@filter_jit
def _qgamma(
    q: Union[Float, ArrayLike],
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
):
    return lax.div(tfp_math.igammainv(shape, q), rate)


def qgamma(
    p: Union[Float, ArrayLike],
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float32,
) -> ArrayLike:
    """Computes the quantile of the gamma distribution.

    Args:
        p: A float or array-like object representing the quantile.
        shape: A float or array-like object representing the shape parameter of the gamma distribution.
        rate: A float or array-like object representing the rate parameter of the gamma distribution.
        lower_tail: A boolean indicating whether to compute the lower tail (default) or upper tail.
        log_prob: A boolean indicating whether to compute the log probability (default False).
        dtype: The dtype of the output. Defaults to float32.

    Returns:
        ArrayLike: The quantile of the gamma distribution.

    Example:
        >>> qgamma(0.5, 0.5, 0.5)
        Array([0.45493677], dtype=float32)
    """
    p = jnp.asarray(p, dtype=dtype)
    p = jnp.atleast_1d(p)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.exp(p)
    x = filter_vmap(_qgamma)(p, shape, rate)
    return x


def rgamma(
    key,
    sample_shape: Optional[Shape] = None,
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float32,
) -> ArrayLike:
    """Generates random gamma values.

    Args:
        key: A PRNGKey to use for the random number generation.
        sample_shape: An optional shape for the output array.
        shape: The shape parameter of the gamma distribution.
        rate: The rate parameter of the gamma distribution.
        lower_tail: Whether to return the lower tail of the distribution.
        log_prob: Whether to return the log probability of the result.
        dtype: The dtype of the output. Defaults to float32.

    Returns:
        ArrayLike: A random gamma value or an array of random gamma values.

    Example:
        >>> rgamma(key, shape=0.5, rate=0.5)
        Array(0.3384059, dtype=float32)
    """
    rvs = _rgamma(key, shape, rate, sample_shape, dtype=dtype)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs


@filter_jit
def _rgamma(
    key: KeyArray,
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
    dtype=jnp.float32,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(shape), jnp.shape(rate))
    shape = jnp.broadcast_to(shape, sample_shape)
    rate = jnp.broadcast_to(rate, sample_shape)
    return jrand.gamma(key, shape, sample_shape, dtype) / rate
