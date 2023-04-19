from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax import lax
from jax._src.random import KeyArray, Shape
from jaxtyping import ArrayLike, Float


@filter_jit
def _pweibull(
    x: Union[Float, ArrayLike],
    concentration: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
):
    concentration = lax.convert_element_type(concentration, x.dtype)
    scale = lax.convert_element_type(scale, x.dtype)
    scaled_x = lax.div(x, scale)
    powered = jnp.float_power(scaled_x, concentration)
    return 1 - jnp.exp(-powered)


def pweibull(
    q: Union[Float, ArrayLike],
    concentration: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
) -> ArrayLike:
    """Computes the cumulative distribution function of the Weibull distribution.

    Args:
        q (Union[Float, ArrayLike]): The value at which to evaluate the CDF.
        concentration (Union[Float, ArrayLike], optional): The concentration parameter of the Weibull distribution. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter of the Weibull distribution. Defaults to 1.0.
        lower_tail (bool, optional): Whether to compute the lower tail of the CDF. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        Array: The cumulative distribution function of the Weibull distribution evaluated at `q`.
    """
    q = jnp.asarray(q)
    q = jnp.atleast_1d(q)
    p = filter_vmap(_pweibull)(q, concentration, scale)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


_dweibull = filter_grad(filter_jit(_pweibull))


def dweibull(
    x, concentration=0.0, scale=1.0, lower_tail=True, log_prob=False
) -> ArrayLike:
    """Computes the probability density function of the Weibull distribution.

    Args:
        x (Union[Float, ArrayLike]): The value at which to evaluate the PDF.
        concentration (Union[Float, ArrayLike], optional): The concentration parameter of the Weibull distribution. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter of the Weibull distribution. Defaults to 1.0.
        lower_tail (bool, optional): Whether to compute the lower tail of the CDF. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        Array: The probability density function of the Weibull distribution evaluated at `x`.
    """
    x = jnp.asarray(x)
    x = jnp.atleast_1d(x)
    grads = filter_vmap(_dweibull)(x, concentration, scale)
    if not lower_tail:
        grads = 1 - grads
    if log_prob:
        grads = jnp.log(grads)
    return grads


@filter_jit
def _qweibull(
    q: Union[Float, ArrayLike],
    concentration: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
) -> ArrayLike:
    concentration = lax.convert_element_type(concentration, q.dtype)
    scale = lax.convert_element_type(scale, q.dtype)
    one = lax.convert_element_type(1, q.dtype)
    nlog_q = -lax.log(lax.sub(one, q))
    inv_concentration = lax.div(one, concentration)
    powerd = jnp.float_power(nlog_q, inv_concentration)
    x = lax.mul(powerd, scale)
    return x


def qweibull(
    p: Union[Float, ArrayLike],
    concentration: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
) -> ArrayLike:
    """Computes the quantile function of the Weibull distribution.

    Args:
        p (Union[Float, ArrayLike]): The quantiles to compute.
        concentration (Union[Float, ArrayLike], optional): The concentration parameter of the Weibull distribution. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter of the Weibull distribution. Defaults to 1.0.
        lower_tail (bool, optional): Whether to compute the lower tail of the distribution. Defaults to True.
        log_prob (bool, optional): Whether to compute the log probability of the distribution. Defaults to False.

    Returns:
        Array: The computed quantiles.
    """
    p = jnp.asarray(p)
    p = jnp.atleast_1d(p)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.exp(p)
    q = filter_vmap(_qweibull)(p, concentration, scale)
    return q


@filter_jit
def _rweibull(
    key: KeyArray,
    concentration: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
):
    return jrand.weibull_min(key, scale, concentration, sample_shape)


def rweibull(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    concentration: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
) -> ArrayLike:
    """Generates samples from the Weibull distribution.

    Args:
        key (KeyArray): Random key used for generating random numbers.
        sample_shape (Optional[Shape], optional): Shape of the output sample. Defaults to None.
        concentration (Union[Float, ArrayLike], optional): Concentration parameter of the Weibull distribution. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): Scale parameter of the Weibull distribution. Defaults to 1.0.
        lower_tail (bool, optional): Whether to return the lower tail probability. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        rvs (ArrayLike): Probability of the Weibull distribution.
    """
    rvs = _rweibull(key, concentration, scale, sample_shape)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
