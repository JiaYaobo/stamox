from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax._src.random import KeyArray, Shape
from jaxtyping import Array, ArrayLike, Bool, Float

from ..core import make_partial_pipe


@filter_jit
def _punif(
    x: Union[Float, ArrayLike],
    mini: Union[Float, Array] = 0.0,
    maxi: Union[Float, Array] = 1.0,
) -> Array:
    p = (x - mini) / (maxi - mini)
    p = jnp.clip(p, a_min=mini, a_max=maxi)
    return p


@make_partial_pipe
def punif(
    q: Union[Float, ArrayLike],
    mini: Union[Float, Array] = 0.0,
    maxi: Union[Float, Array] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> Array:
    """Computes the cumulative distribution function of the uniform distribution.

    Args:
        q (Union[Float, ArrayLike]): The value at which to evaluate the CDF.
        mini (Union[Float, ArrayLike], optional): The minimum value of the uniform distribution. Defaults to 0.0.
        maxi (Union[Float, ArrayLike], optional): The maximum value of the uniform distribution. Defaults to 1.0.
        lower_tail (bool, optional): Whether to compute the lower tail of the CDF. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        Array: The cumulative distribution function of the uniform distribution evaluated at `q`.
    """
    q = jnp.atleast_1d(q)
    p = filter_vmap(_punif)(q, mini, maxi)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


_dunif = filter_grad(filter_jit(_punif))


@make_partial_pipe
def dunif(
    x: Union[Float, ArrayLike],
    mini: Union[Float, Array] = 0.0,
    maxi: Union[Float, Array] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
):
    """Calculates the probability density function of a uniform distribution.

    Args:
        x (Union[Float, ArrayLike]): The value or array of values for which to calculate the probability density.
        mini (Union[Float, Array], optional): The lower bound of the uniform distribution. Defaults to 0.0.
        maxi (Union[Float, Array], optional): The upper bound of the uniform distribution. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to calculate the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        float or ndarray: The probability density of the given value(s).
    """
    x = jnp.atleast_1d(x)
    p = filter_vmap(_dunif)(x, mini, maxi)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


@filter_jit
def _qunif(
    q: Union[Float, ArrayLike],
    mini: Union[Float, Array] = 0.0,
    maxi: Union[Float, Array] = 1.0,
):
    x = q * (maxi - mini) + mini
    x = jnp.clip(x, a_min=mini, a_max=maxi)
    return x


@make_partial_pipe
def qunif(
    p: Union[Float, ArrayLike],
    mini: Union[Float, Array] = 0.0,
    maxi: Union[Float, Array] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
):
    """
    Computes the quantile function of a uniform distribution.

    Args:
        p (Union[Float, ArrayLike]): Quantiles to compute.
        mini (Union[Float, Array], optional): Lower bound of the uniform distribution. Defaults to 0.0.
        maxi (Union[Float, Array], optional): Upper bound of the uniform distribution. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to compute the lower tail or not. Defaults to True.
        log_prob (Bool, optional): Whether to compute the log probability or not. Defaults to False.

    Returns:
        Union[Float, Array]: The quantiles of the uniform distribution.
    """
    p = jnp.atleast_1d(p)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.exp(p)
    q = filter_vmap(_qunif)(p, mini, maxi)
    return q


@filter_jit
def _runif(
    key: KeyArray,
    mini: Union[Float, Array] = 0.0,
    maxi: Union[Float, Array] = 1.0,
    sample_shape: Optional[Shape] = None,
):
    return jrand.uniform(key, sample_shape, minval=mini, maxval=maxi)


@make_partial_pipe
def runif(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    mini: Union[Float, Array] = 0.0,
    maxi: Union[Float, Array] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
):
    """Generates random numbers from a uniform distribution.

    Args:
        key: A PRNGKey to use for generating the random numbers.
        sample_shape: The shape of the output array.
        mini: The minimum value of the uniform distribution.
        maxi: The maximum value of the uniform distribution.
        lower_tail: Whether to generate values from the lower tail of the distribution.
        log_prob: Whether to return the log probability of the generated values.

    Returns:
        An array of random numbers from a uniform distribution.
    """
    rvs = _runif(key, mini, maxi, sample_shape)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
