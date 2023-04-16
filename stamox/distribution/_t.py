from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax import lax
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
    df = lax.convert_element_type(df, x.dtype)
    loc = lax.convert_element_type(loc, x.dtype)
    scale = lax.convert_element_type(scale, x.dtype)
    scaled = lax.div(lax.sub(x, loc), scale)
    scaled_squared = lax.integer_pow(scaled, 2)
    beta_value = lax.div(df, lax.add(df, scaled_squared))
    return 0.5 * (
        1. + jnp.sign(scaled) - jnp.sign(scaled) * betainc(0.5 * df, 0.5, beta_value)
    )


@make_partial_pipe
def pt(
    q: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float32,
) -> ArrayLike:
    """Calculates the probability of a given value for Student T distribution.

    Args:
        q: The value to calculate the probability of.
        df: The degrees of freedom of the distribution.
        loc: The location parameter of the distribution.
        scale: The scale parameter of the distribution.
        lower_tail: Whether to calculate the lower tail probability or not.
        log_prob: Whether to return the log probability or not.
        dtype: The dtype of the output. Defaults to jnp.float32.

    Returns:
        ArrayLike: The cdf value of the given value for Student T distribution.

    Example:
        >>> pt(1.0, 1.0)
        Array([0.74999994], dtype=float32, weak_type=True)
    """
    q = jnp.asarray(q, dtype=dtype)
    q = jnp.atleast_1d(q)
    p = filter_vmap(_pt)(q, df, loc, scale)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)

    return p


_dt = filter_jit(filter_grad(_pt))


@make_partial_pipe
def dt(
    x: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float32,
) -> ArrayLike:
    """Calculates the probability density function of a Student's t-distribution.

    Args:
        x: A float or array-like object representing the values at which to evaluate
        the probability density function.
        df: Degrees of freedom for the Student's t-distribution.
        loc: Location parameter for the Student's t-distribution. Defaults to 0.0.
        scale: Scale parameter for the Student's t-distribution. Defaults to 1.0.
        lower_tail: Whether to return the lower tail probability. Defaults to True.
        log_prob: Whether to return the log probability. Defaults to False.
        dtype: The dtype of the output. Defaults to jnp.float32.

    Returns:
        ArrayLike: The probability density function evaluated at `x`.

    Example:
        >>> dt(1.0, 1.0)
        Array([0.1591549], dtype=float32, weak_type=True)
    """
    x = jnp.asarray(x, dtype=dtype)
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
    df = lax.convert_element_type(df, q.dtype)
    loc = lax.convert_element_type(loc, q.dtype)
    scale = lax.convert_element_type(scale, q.dtype)
    one = lax.convert_element_type(1, q.dtype)
    beta_value = tfp_special.betaincinv(0.5 * df, 0.5, 1 - jnp.abs(1 - 2 * q))
    scaled_squared = df * (one / beta_value - one)
    scaled = jnp.sign(q - 0.5) * jnp.sqrt(scaled_squared)
    return scaled * scale + loc


@make_partial_pipe
def qt(
    p: Union[Float, ArrayLike],
    df: Union[Int, Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float32,
) -> ArrayLike:
    """Calculates the quantile of Student T distribution.

    Args:
        p: A float or array-like object representing the quantile to be calculated.
        df: An int, float, or array-like object representing the degrees of freedom.
        loc: An optional float or array-like object representing the location parameter. Defaults to 0.0.
        scale: An optional float or array-like object representing the scale parameter. Defaults to 1.0.
        lower_tail: A boolean indicating whether the lower tail should be used. Defaults to True.
        log_prob: A boolean indicating whether the probability should be logged. Defaults to False.
        dtype: The dtype of the output. Defaults to jnp.float32.

    Returns:
        ArrayLike: The quantile of the Student T distribution.

    Example:
        >>> qt(0.5, 1.0)
        Array([0.], dtype=float32, weak_type=True)
    """
    p = jnp.asarray(p, dtype=dtype)
    p = jnp.atleast_1d(p)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.exp(p)
    q = filter_vmap(_qt)(p, df, loc, scale)
    return q


@filter_jit
def _rt(
    key: KeyArray,
    df: Union[Int, Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
    dtype=jnp.float32,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(
            jnp.shape(df), jnp.shape(loc), jnp.shape(scale)
        )
    scale = jnp.broadcast_to(scale, sample_shape)
    loc = jnp.broadcast_to(loc, sample_shape)
    return jrand.t(key, df, sample_shape, dtype=dtype) * scale + loc


@make_partial_pipe
def rt(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    df: Union[Int, Float, ArrayLike] = None,
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float32,
) -> ArrayLike:
    """Generates random numbers from a t-distribution.

    Args:
        key: Type of the random number generator.
        sample_shape: Shape of the output array.
        df: Degrees of freedom.
        loc: Location parameter.
        scale: Scale parameter.
        lower_tail: Whether to return the lower tail probability. Defaults to True.
        log_prob: Whether to return the log probability. Defaults to False.
        dtype: The dtype of the output. Defaults to jnp.float32.

    Returns:
        ArrayLike: Random numbers from a t-distribution.

    Example:
        >>> rt(key, (2, 3), 1.0)
        Array([[1.9982358e+02, 2.3699088e-01, 6.6509140e-01],
                [5.3681795e-02, 3.3967651e+01, 6.8611817e+00]], dtype=float32)
    """
    rvs = _rt(key, df, loc, scale, sample_shape, dtype=dtype)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
