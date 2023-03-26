from typing import Union, Optional

import jax.numpy as jnp
import jax.random as jrand
from jax.random import KeyArray
from jax._src.random import Shape
from equinox import filter_jit, filter_grad, filter_vmap
from jaxtyping import ArrayLike, Float, Array, Bool

from ..core import make_partial_pipe


@filter_jit
def _plaplace(
    x: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
) -> Array:
    scaled = (x - loc) / scale
    return 0.5 - 0.5 * jnp.sign(scaled) * jnp.expm1(-jnp.abs(scaled))


@make_partial_pipe
def plaplace(
    x: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> Array:
    """Calculates the Laplace cumulative density function.

    Args:
        x (Union[Float, ArrayLike]): The value at which to evaluate the Plaplace PDF.
        loc (Union[Float, ArrayLike], optional): The location parameter of the Plaplace PDF. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter of the Plaplace PDF. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to return the lower tail of the Plaplace PDF. Defaults to True.
        log_prob (Bool, optional): Whether to return the logarithm of the Plaplace PDF. Defaults to False.

    Returns:
        Array: The Laplace CDF evaluated at `x`.
    """
    x = jnp.atleast_1d(x)
    p = filter_vmap(_plaplace)(x, loc, scale)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


_dlaplace = filter_grad(filter_jit(_plaplace))


@make_partial_pipe
def dlaplace(
    x: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> Array:
    """Calculates the Laplace probability density function for a given x, location and scale.

    Args:
        x (Union[Float, ArrayLike]): The value at which to calculate the probability density.
        loc (Union[Float, ArrayLike], optional): The location parameter. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to return the lower tail of the distribution. Defaults to True.
        log_prob (Bool, optional): Whether to return the logarithm of the probability. Defaults to False.

    Returns:
        Array: The probability density at the given x.
    """
    x = jnp.atleast_1d(x)
    p = filter_vmap(_dlaplace)(x, loc, scale)
    if not lower_tail:
        p = -p
    if log_prob:
        p = jnp.log(p)
    return p


@filter_jit
def _qlaplace(
    q: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
):
    a = q - 0.5
    return loc - scale * jnp.sign(a) * jnp.log1p(-2 * jnp.abs(a))


@make_partial_pipe
def qlaplace(
    q: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> Array:
    """Computes the quantile of the Laplace distribution.

    Args:
        q (Union[Float, ArrayLike]): Quantiles to compute.
        loc (Union[Float, ArrayLike], optional): Location parameter. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): Scale parameter. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to compute the lower tail. Defaults to True.
        log_prob (Bool, optional): Whether to compute the log probability. Defaults to False.

    Returns:
        Array: The quantiles of the Laplace distribution.
    """
    q = jnp.atleast_1d(q)
    if not lower_tail:
        q = 1 - q
    if log_prob:
        q = jnp.exp(q)
    return filter_vmap(_qlaplace)(q, loc, scale)


@filter_jit
def _rlaplace(
    key: KeyArray,
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    loc = jnp.broadcast_to(loc, sample_shape)
    scale = jnp.broadcast_to(scale, sample_shape)
    return jrand.laplace(key, sample_shape) * scale + loc


@make_partial_pipe
def rlaplace(
    key: KeyArray,
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> Array:
    """Generates random Laplace samples from a given key.

    Args:
        key (KeyArray): The PRNG key to use for generating the samples.
        loc (Union[Float, ArrayLike], optional): The location parameter of the Laplace distribution. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter of the Laplace distribution. Defaults to 1.0.
        sample_shape (Optional[Shape], optional): The shape of the output array. Defaults to None.
        lower_tail (Bool, optional): Whether to return the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        Array: An array containing the random Laplace samples.
    """
    probs = _rlaplace(key, loc, scale, sample_shape)
    if not lower_tail:
        probs = 1 - probs
    if log_prob:
        probs = jnp.log(probs)
    return probs
