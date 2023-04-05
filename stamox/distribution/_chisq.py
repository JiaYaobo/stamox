from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_jit, filter_vmap
from jax._src.random import KeyArray, Shape
from jaxtyping import ArrayLike, Float, Int

from ..core import make_partial_pipe
from ._gamma import _dgamma, _pgamma, _qgamma


@make_partial_pipe
def dchisq(
    x: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
):
    """Computes the chi-squared distribution.

    Args:
      x: A float or array-like object representing the values at which to
        evaluate the chi-squared distribution.
      df: The degrees of freedom for the chi-squared distribution.
      lower_tail: A boolean indicating whether to compute the lower tail of the
        chi-squared distribution (defaults to True).
      log_prob: A boolean indicating whether to return the log probability
        (defaults to False).

    Returns:
      The chi-squared distribution evaluated at `x`.
    """
    x = jnp.atleast_1d(x)
    grads = filter_vmap(_dgamma)(x, df / 2, 1 / 2)
    if not lower_tail:
        grads = 1 - grads
    if log_prob:
        grads = jnp.log(grads)
    return grads


@make_partial_pipe
def pchisq(
    q: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
):
    """Calculates the chi-squared probability density function.

    Args:
        q: A float or array-like object representing the value of the chi-squared
        variable.
        df: An int, float, or array-like object representing the degrees of freedom.
        lower_tail: A boolean indicating whether to calculate the lower tail (default
        True).
        log_prob: A boolean indicating whether to return the log probability (default
        False).

    Returns:
        The chi-squared probability density function.
    """
    q = jnp.atleast_1d(q)
    p = filter_vmap(_pgamma)(q, df / 2, 1 / 2)
    if not lower_tail:
        p = 1.0 - p
    if log_prob:
        p = jnp.log(p)
    return p


@make_partial_pipe
def qchisq(
    p: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
):
    """Computes the quantile of the chi-squared distribution.

    Args:
      p: A float or array-like object representing the quantile.
      df: An int, float, or array-like object representing the degrees of freedom.
      lower_tail: A boolean indicating whether to compute the lower tail (defaults to True).
      log_prob: A boolean indicating whether to compute the log probability (defaults to False).

    Returns:
      The quantile of the chi-squared distribution.
    """
    p = jnp.atleast_1d(p)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.exp(p)
    q = filter_vmap(_qgamma)(p, df / 2, 1 / 2)
    return q


@filter_jit
def _rchisq(
    key: KeyArray,
    df: Union[Int, Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
):
    if sample_shape is None:
        sample_shape = jnp.shape(df)
    df = jnp.broadcast_to(df, sample_shape)
    return jrand.chisquare(key, df, shape=sample_shape)


@make_partial_pipe
def rchisq(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    df: Union[Int, Float, ArrayLike] = None,
    lower_tail=True,
    log_prob=False,
):
    """Generates random chisquare-distribution numbers.

    Args:
      key: A KeyArray object.
      sample_shape: An optional shape for the sample.
      df: An integer, float, or array-like object.
      lower_tail: A boolean indicating whether to use the lower tail of the distribution.
      log_prob: A boolean indicating whether to return the log probability.

    Returns:
      rvs: The random chisquare-distribution numbers.
    """
    rvs = _rchisq(key, df, sample_shape)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
