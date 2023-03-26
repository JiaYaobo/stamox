from typing import Union, Optional

import jax.random as jrand
import jax.numpy as jnp
from jax._src.random import Shape, KeyArray
from equinox import filter_jit, filter_grad, filter_vmap
from jaxtyping import ArrayLike, Float, Int

from ._gamma import _pgamma, _qgamma, _rgamma, _dgamma
from ..core import make_partial_pipe


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
    x: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
):
    """Calculates the chi-squared probability density function.

    Args:
        x: A float or array-like object representing the value of the chi-squared
        variable.
        df: An int, float, or array-like object representing the degrees of freedom.
        lower_tail: A boolean indicating whether to calculate the lower tail (default
        True).
        log_prob: A boolean indicating whether to return the log probability (default
        False).

    Returns:
        The chi-squared probability density function.
    """
    x = jnp.atleast_1d(x)
    p = filter_vmap(_pgamma)(x, df / 2, 1 / 2)
    if not lower_tail:
        p = 1.0 - p
    if log_prob:
        p = jnp.log(p)
    return p


@make_partial_pipe
def qchisq(
    q: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
):
    """Computes the quantile of the chi-squared distribution.

    Args:
      q: A float or array-like object representing the quantile.
      df: An int, float, or array-like object representing the degrees of freedom.
      lower_tail: A boolean indicating whether to compute the lower tail (defaults to True).
      log_prob: A boolean indicating whether to compute the log probability (defaults to False).

    Returns:
      The quantile of the chi-squared distribution.
    """
    q = jnp.atleast_1d(q)
    if not lower_tail:
        q = 1 - q
    if log_prob:
        q = jnp.exp(q)
    x = filter_vmap(_qgamma)(q, df / 2, 1 / 2)
    return x


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
    df: Union[Int, Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
    lower_tail=True,
    log_prob=False,
):
    """Generates random chisquare-distribution numbers.

    Args:
      key: A KeyArray object.
      df: An integer, float, or array-like object.
      sample_shape: An optional shape for the sample.
      lower_tail: A boolean indicating whether to use the lower tail of the distribution.
      log_prob: A boolean indicating whether to return the log probability.

    Returns:
      rv: The random chisquare-distribution numbers.
    """
    rv = _rchisq(key, df, sample_shape)
    if not lower_tail:
        rv = 1 - rv
    if log_prob:
        rv = jnp.log(rv)
    return rv
