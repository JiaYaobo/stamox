from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax import lax
from jax._src.random import KeyArray, Shape
from jaxtyping import ArrayLike, Float
from tensorflow_probability.substrates.jax.math import special as tfp_special

from ._utils import (
    _check_clip_distribution_domain,
    _check_clip_probability,
    _post_process,
    _promote_dtype_to_floating,
)


@filter_jit
def _pbeta(
    x: Union[Float, ArrayLike], a: Union[Float, ArrayLike], b: Union[Float, ArrayLike]
):
    dtype = lax.dtype(x)
    a = jnp.asarray(a, dtype=dtype)
    b = jnp.asarray(b, dtype=dtype)
    return tfp_special.betainc(a, b, x)


def pbeta(
    q: Union[Float, ArrayLike],
    a: Union[Float, ArrayLike],
    b: Union[Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the cumulative distribution function of the beta distribution.

    Args:
        q (Union[Float, ArrayLike]): Quantiles.
        a (Union[Float, ArrayLike]): Shape parameter.
        b (Union[Float, ArrayLike]): Shape parameter.
        lower_tail (bool, optional): If True (default), probabilities are P[X ≤ x], otherwise, P[X > x].
        log_prob (bool, optional): If True, probabilities are given as log(P).
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to None.

    Returns:
        ArrayLike: The probability or log of the probability for each quantile.

    Example:
        >>> q = jnp.array([0.1, 0.5, 0.9])
        >>> a = 2.0
        >>> b = 3.0
        >>> pbeta(q, a, b)
        Array([0.05230004, 0.68749976, 0.9963    ], dtype=float32)
    """
    q, dtype = _promote_dtype_to_floating(q, dtype)
    q = jnp.atleast_1d(q)
    q = _check_clip_distribution_domain(q, 0.0, 1.0)
    p = filter_vmap(_pbeta)(q, a, b)
    p = _post_process(p, lower_tail, log_prob)
    return p


_dbeta = filter_jit(filter_grad(_pbeta))


def dbeta(
    x: Union[Float, ArrayLike],
    a: Union[Float, ArrayLike],
    b: Union[Float, ArrayLike],
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculates the probability density function of the beta distribution.

    Args:
      x: A float or array-like object representing the value(s) at which to evaluate the PDF.
      a: A float or array-like object representing the shape parameter of the beta distribution.
      b: A float or array-like object representing the scale parameter of the beta distribution.
      lower_tail: A boolean indicating whether to calculate the lower tail (default True).
      log_prob: A boolean indicating whether to return the logarithm of the PDF (default False).
      dtype: The dtype of the output. Defaults to None.

    Returns:
      ArrayLike: The probability density function of the beta distribution evaluated at x.

    Example:
        >>> dbeta(0.5, 2, 3, lower_tail=True, log_prob=False)
        Array([1.4999996], dtype=float32, weak_type=True)
    """
    x, dtype = _promote_dtype_to_floating(x, dtype)
    x = jnp.atleast_1d(x)
    x = _check_clip_distribution_domain(x, 0.0, 1.0)
    d = filter_vmap(_dbeta)(x, a, b)
    d = _post_process(d, lower_tail, log_prob)
    return d


@filter_jit
def _qbeta(
    p: Union[Float, ArrayLike], a: Union[Float, ArrayLike], b: Union[Float, ArrayLike]
):
    return tfp_special.betaincinv(a, b, p)


def qbeta(
    p: Union[Float, ArrayLike],
    a: Union[Float, ArrayLike],
    b: Union[Float, ArrayLike],
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the quantile of beta distribution function.

    Args:
        p: A float or array-like object representing the quantile.
        a: A float or array-like object representing the alpha parameter.
        b: A float or array-like object representing the beta parameter.
        lower_tail: A boolean indicating whether to compute the lower tail of the
        distribution (defaults to True).
        log_prob: A boolean indicating whether to compute the log probability
        (defaults to False).
        dtype: The dtype of the output. Defaults to None.

    Returns:
        ArrayLike: The value of the beta distribution at the given quantile.

    Example:
        >>> qbeta(0.5, 2, 3, lower_tail=True, log_prob=False)
        Array([0.38572744], dtype=float32)
    """
    p, dtype = _promote_dtype_to_floating(p, dtype)
    p = jnp.atleast_1d(p)
    p = _check_clip_probability(p, lower_tail=lower_tail, log_prob=log_prob)
    x = filter_vmap(_qbeta)(p, a, b)
    return x


def rbeta(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    a: Union[Float, ArrayLike] = None,
    b: Union[Float, ArrayLike] = None,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Generates random numbers from the Beta distribution.

    Args:
        key: A PRNGKey used for random number generation.
        sample_shape: An optional shape for the output samples.
        a: The shape parameter of the Beta distribution. Can be either a float or an array-like object.
        b: The scale parameter of the Beta distribution. Can be either a float or an array-like object.
        lower_tail: Whether to return the lower tail probability (defaults to True).
        log_prob: Whether to return the log probability (defaults to False).
        dtype: The dtype of the output. Defaults to jnp.float32.

    Returns:
        ArrayLike: Random numbers from the Beta distribution.

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> rbeta(key, sample_shape=(3,), a=2, b=3)
        Array([0.02809353, 0.13760717, 0.49360353], dtype=float32)
    """
    rvs = _rbeta(key, a, b, sample_shape, dtype=dtype)
    rvs = _post_process(rvs, lower_tail, log_prob)
    return rvs


@filter_jit
def _rbeta(
    key: KeyArray,
    a: Union[Float, ArrayLike],
    b: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
    dtype=jnp.float_,
) -> ArrayLike:
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(a), jnp.shape(b))
    a = jnp.broadcast_to(a, sample_shape)
    b = jnp.broadcast_to(b, sample_shape)
    return jrand.beta(key, a, b, sample_shape, dtype=dtype)
