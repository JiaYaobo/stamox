from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax._src.random import KeyArray, Shape
from jax.scipy.special import betainc
from jaxtyping import ArrayLike, Float
from tensorflow_probability.substrates.jax.math import special as tfp_special

from ..core import make_partial_pipe


@filter_jit
def _pbeta(
    x: Union[Float, ArrayLike], a: Union[Float, ArrayLike], b: Union[Float, ArrayLike]
):
    return betainc(a, b, x)


@make_partial_pipe
def pbeta(
    x: Union[Float, ArrayLike],
    a: Union[Float, ArrayLike],
    b: Union[Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
):
    x = jnp.atleast_1d(x)
    p = filter_vmap(_pbeta)(x, a, b)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


_dbeta = filter_jit(filter_grad(_pbeta))


@make_partial_pipe
def dbeta(
    x: Union[Float, ArrayLike],
    a: Union[Float, ArrayLike],
    b: Union[Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
):
    """Calculates the probability density function of the beta distribution.

    Args:
      x: A float or array-like object representing the value(s) at which to evaluate the PDF.
      a: A float or array-like object representing the shape parameter of the beta distribution.
      b: A float or array-like object representing the scale parameter of the beta distribution.
      lower_tail: A boolean indicating whether to calculate the lower tail (default True).
      log_prob: A boolean indicating whether to return the logarithm of the PDF (default False).

    Returns:
      The probability density function of the beta distribution evaluated at x.
    """
    x = jnp.atleast_1d(x)
    p = filter_vmap(_dbeta)(x, a, b)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


@filter_jit
def _qbeta(
    q: Union[Float, ArrayLike], a: Union[Float, ArrayLike], b: Union[Float, ArrayLike]
):
    return tfp_special.betaincinv(a, b, q)


@make_partial_pipe
def qbeta(
    q: Union[Float, ArrayLike],
    a: Union[Float, ArrayLike],
    b: Union[Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
):
    """Computes the beta distribution function.

    Args:
      q: A float or array-like object representing the quantile.
      a: A float or array-like object representing the alpha parameter.
      b: A float or array-like object representing the beta parameter.
      lower_tail: A boolean indicating whether to compute the lower tail of the
        distribution (defaults to True).
      log_prob: A boolean indicating whether to compute the log probability
        (defaults to False).

    Returns:
      The value of the beta distribution at the given quantile.
    """
    q = jnp.atleast_1d(q)
    if not lower_tail:
        q = 1 - q
    if log_prob:
        q = jnp.exp(q)
    x = filter_vmap(_qbeta)(q, a, b)
    return x


@make_partial_pipe
def rbeta(
    key: KeyArray,
    a: Union[Float, ArrayLike],
    b: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
    lower_tail=True,
    log_prob=False,
):
    """Generates random numbers from the Beta distribution.

    Args:
        key: A PRNGKey used for random number generation.
        a: The shape parameter of the Beta distribution. Can be either a float or an array-like object.
        b: The scale parameter of the Beta distribution. Can be either a float or an array-like object.
        sample_shape: An optional shape for the output samples.
        lower_tail: Whether to return the lower tail probability (defaults to True).
        log_prob: Whether to return the log probability (defaults to False).

    Returns:
        A numpy array containing random numbers from the Beta distribution.
    """
    probs = _rbeta(key, a, b, sample_shape)
    if not lower_tail:
        probs = 1 - probs
    if log_prob:
        probs = jnp.log(probs)
    return probs


@filter_jit
def _rbeta(
    key: KeyArray,
    a: Union[Float, ArrayLike],
    b: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(a), jnp.shape(b))
    a = jnp.broadcast_to(a, sample_shape)
    b = jnp.broadcast_to(b, sample_shape)
    return jrand.beta(key, a, b, sample_shape)
