from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax._src.random import Shape
from jax.random import KeyArray
from jax.scipy.special import betainc
from jaxtyping import ArrayLike, Float, Int
from tensorflow_probability.substrates.jax.math import special as tfp_special

from ..core import make_partial_pipe


@filter_jit
def _pt(
    x: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
):
    scaled = (x - loc) / scale
    scaled_squared = scaled * scaled
    beta_value = df / (df + scaled_squared)
    return 0.5 * (
        1 + jnp.sign(scaled) - jnp.sign(scaled) * betainc(0.5 * df, 0.5, beta_value)
    )


@make_partial_pipe(name='pt')
def pt(
    x,
    df: Union[Int, Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
):
    """Calculates the probability of a given value for Student T distribution.

    Args:
        x: The value to calculate the probability of.
        df: The degrees of freedom of the distribution.
        loc: The location parameter of the distribution.
        scale: The scale parameter of the distribution.
        lower_tail: Whether to calculate the lower tail probability or not.
        log_prob: Whether to return the log probability or not.

    Returns:
        The probability of the given value for Student T distribution.
    """
    x = jnp.atleast_1d(x)
    p = filter_vmap(_pt)(x, df, loc, scale)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)

    return p


_dt = filter_jit(filter_grad(_pt))


@make_partial_pipe(name='dt')
def dt(
    x: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
):
    """Calculates the probability density function of a Student's t-distribution.

    Args:
        x: A float or array-like object representing the values at which to evaluate
        the probability density function.
        df: Degrees of freedom for the Student's t-distribution.
        loc: Location parameter for the Student's t-distribution. Defaults to 0.0.
        scale: Scale parameter for the Student's t-distribution. Defaults to 1.0.
        lower_tail: Whether to return the lower tail probability. Defaults to True.
        log_prob: Whether to return the log probability. Defaults to False.

    Returns:
        The probability density function evaluated at `x`.
    """
    x = jnp.atleast_1d(x)
    grads = filter_vmap(_dt)(x, df, loc, scale)
    if not lower_tail:
        grads = 1 - grads
    if log_prob:
        grads = jnp.log(grads)
    return grads


@filter_jit
def _qt(
    q: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
):
    beta_value = tfp_special.betaincinv(0.5 * df, 0.5, 1 - jnp.abs(1 - 2 * q))
    scaled_squared = df * (1 / beta_value - 1)
    scaled = jnp.sign(q - 0.5) * jnp.sqrt(scaled_squared)
    return scaled * scale + loc


@make_partial_pipe(name='qt')
def qt(
    q: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
):
    """Calculates the quantile of Student T distribution.

    Args:
        q: A float or array-like object representing the quantile to be calculated.
        df: An int, float, or array-like object representing the degrees of freedom.
        loc: An optional float or array-like object representing the location parameter. Defaults to 0.0.
        scale: An optional float or array-like object representing the scale parameter. Defaults to 1.0.
        lower_tail: A boolean indicating whether the lower tail should be used. Defaults to True.
        log_prob: A boolean indicating whether the probability should be logged. Defaults to False.

    Returns:
        The quantile of the Student T distribution.
    """
    q = jnp.atleast_1d(q)
    if not lower_tail:
        q = 1 - q
    if log_prob:
        q = jnp.exp(q)
    q = filter_vmap(_qt)(q, df, loc, scale)
    return q


@filter_jit
def _rt(
    key: KeyArray,
    df: Union[Int, Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(df), jnp.shape(loc), jnp.shape(scale))
    scale = jnp.broadcast_to(scale, sample_shape)
    loc = jnp.broadcast_to(loc, sample_shape)
    return jrand.t(key, df, sample_shape) * scale + loc

@make_partial_pipe(name='rt')
def rt(
    key: KeyArray,
    df: Union[Int, Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
    lower_tail = True,
    log_prob = False,
):
    """Generates random numbers from a t-distribution.

    Args:
        key: Type of the random number generator.
        df: Degrees of freedom.
        loc: Location parameter.
        scale: Scale parameter.
        sample_shape: Shape of the output array.

    Returns:
        Random numbers from a t-distribution.
    """
    probs = _rt(key, df, loc, scale, sample_shape)
    if not lower_tail:
        probs =  1 - probs
    if log_prob:
        probs = jnp.log(probs)
    return probs
