from typing import Union, Optional

import jax.random as jrand
import jax.numpy as jnp
from jax.scipy.special import gammainc
from jax._src.random import Shape, KeyArray
from equinox import filter_jit, filter_grad, filter_vmap
import tensorflow_probability.substrates.jax.math as tfp_math
from jaxtyping import ArrayLike, Float

from ..core import make_partial_pipe


@filter_jit
def _pgamma(
    x, shape: Union[Float, ArrayLike] = 1.0, rate: Union[Float, ArrayLike] = 1.0
):
    """Computes the gamma function for a given x, shape, and rate.

    Args:
        x: A float or array-like object representing the input to the gamma function.
        shape: A float or array-like object representing the shape parameter of the gamma function.
        rate: A float or array-like object representing the rate parameter of the gamma function.

    Returns:
        The result of the gamma function for the given inputs.
    """
    return gammainc(shape, x * rate)


@make_partial_pipe
def pgamma(
    x: Union[Float, ArrayLike],
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
):
    """Computes the probability density function of the gamma distribution.

    Args:
        x: A float or array-like object representing the input to the gamma function.
        shape: A float or array-like object representing the shape parameter of the gamma function.
        rate: A float or array-like object representing the rate parameter of the gamma function.
        lower_tail: A boolean indicating whether to compute the lower tail of the gamma function.
        log_prob: A boolean indicating whether to compute the logarithm of the probability density function.

    Returns:
        The probability density function of the gamma distribution for the given inputs.
    """
    x = jnp.atleast_1d(x)
    p = filter_vmap(_pgamma)(x, shape, rate)
    if not lower_tail:
        p = 1.0 - p
    if log_prob:
        p = jnp.log(p)
    return p


_dgamma = filter_jit(filter_grad(_pgamma))


@make_partial_pipe
def dgamma(
    x: Union[Float, ArrayLike],
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
):
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

    Returns:
        grads (float or array): The density of the gamma distribution evaluated
            at `x`. If `log_prob` is True, returns the log probability.
    """
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
    return tfp_math.igammainv(shape, q) / rate


@make_partial_pipe
def qgamma(
    q: Union[Float, ArrayLike],
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
):
    """Computes the quantile of the gamma distribution.

    Args:
        q: A float or array-like object representing the quantile.
        shape: A float or array-like object representing the shape parameter of the gamma distribution.
        rate: A float or array-like object representing the rate parameter of the gamma distribution.
        lower_tail: A boolean indicating whether to compute the lower tail (default) or upper tail.
        log_prob: A boolean indicating whether to compute the log probability (default False).

    Returns:
        The quantile of the gamma distribution.
    """
    q = jnp.atleast_1d(q)
    if not lower_tail:
        q = 1 - q
    if log_prob:
        q = jnp.exp(q)
    x = filter_vmap(_qgamma)(q, shape, rate)
    return x


@make_partial_pipe
def rgamma(
    key,
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
    lower_tail=True,
    log_prob=False,
):
    """Generates random gamma values.

    Args:
        key: A PRNGKey to use for the random number generation.
        shape: The shape parameter of the gamma distribution.
        rate: The rate parameter of the gamma distribution.
        sample_shape: An optional shape for the output array.
        lower_tail: Whether to return the lower tail of the distribution.
        log_prob: Whether to return the log probability of the result.

    Returns:
        A random gamma value or an array of random gamma values.
    """
    rv = _rgamma(key, shape, rate, sample_shape)
    if not lower_tail:
        rv = 1 - rv
    if log_prob:
        rv = jnp.log(rv)
    return rv


@filter_jit
def _rgamma(
    key: KeyArray,
    shape: Union[Float, ArrayLike] = 1.0,
    rate: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(shape), jnp.shape(rate))
    shape = jnp.broadcast_to(shape, sample_shape)
    rate = jnp.broadcast_to(rate, sample_shape)
    return jrand.gamma(key, shape, sample_shape) / rate
