from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax import lax
from jax._src.random import Shape
from jax.random import KeyArray
from jaxtyping import ArrayLike, Bool, Float

from ._utils import (
    _check_clip_distribution_domain,
    _check_clip_probability,
    _post_process,
    _promote_dtype_to_floating,
)


@filter_jit
def _plaplace(
    x: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
):
    dtype = lax.dtype(x)
    loc = jnp.asarray(loc, dtype=dtype)
    scale = jnp.asarray(scale, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    scaled = lax.div(lax.sub(x, loc), scale)
    subtrahend = half * lax.mul(jnp.sign(scaled), lax.expm1(-lax.abs(scaled)))
    return lax.sub(half, subtrahend)


def plaplace(
    q: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculates the Laplace cumulative density function.

    Args:
        q (Union[Float, ArrayLike]): The value at which to evaluate the Plaplace PDF.
        loc (Union[Float, ArrayLike], optional): The location parameter of the Plaplace PDF. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter of the Plaplace PDF. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to return the lower tail of the Plaplace PDF. Defaults to True.
        log_prob (Bool, optional): Whether to return the logarithm of the Plaplace PDF. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The Laplace CDF evaluated at `q`.

    Example:
        >>> plaplace(1.0, 1.0, 1.0)
        Array([0.5], dtype=float32, weak_type=True)
    """
    q, _ = _promote_dtype_to_floating(q, dtype)
    q = jnp.atleast_1d(q)
    q = _check_clip_distribution_domain(q)
    p = filter_vmap(_plaplace)(q, loc, scale)
    p = _post_process(p, lower_tail, log_prob)
    return p


_dlaplace = filter_grad(filter_jit(_plaplace))


def dlaplace(
    x: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculates the Laplace probability density function for a given x, location and scale.

    Args:
        x (Union[Float, ArrayLike]): The value at which to calculate the probability density.
        loc (Union[Float, ArrayLike], optional): The location parameter. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to return the lower tail of the distribution. Defaults to True.
        log_prob (Bool, optional): Whether to return the logarithm of the probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The probability density at the given x.

    Example:
        >>> dlaplace(1.0, 1.0, 1.0)
        Array([0.], dtype=float32, weak_type=True)
    """
    x, _ = _promote_dtype_to_floating(x, dtype)
    x = jnp.atleast_1d(x)
    x = _check_clip_distribution_domain(x)
    grads = filter_vmap(_dlaplace)(x, loc, scale)
    grads = _post_process(grads, lower_tail, log_prob)
    return grads


@filter_jit
def _qlaplace(
    q: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
):
    dtype = lax.dtype(q)
    loc = jnp.asarray(loc, dtype=dtype)
    scale = jnp.asarray(scale, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    a = lax.sub(half, q)
    return scale * jnp.sign(a) * jnp.log1p(-2 * jnp.abs(a)) - loc


def qlaplace(
    p: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the quantile of the Laplace distribution.

    Args:
        p (Union[Float, ArrayLike]): Quantiles to compute.
        loc (Union[Float, ArrayLike], optional): Location parameter. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): Scale parameter. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to compute the lower tail. Defaults to True.
        log_prob (Bool, optional): Whether to compute the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The quantiles of the Laplace distribution.

    Example:
        >>> qlaplace(0.5, 1.0, 1.0)
        Array([1.], dtype=float32, weak_type=True)
    """
    p, _ = _promote_dtype_to_floating(p, dtype)
    p = jnp.atleast_1d(p)
    p = _check_clip_probability(p, lower_tail, log_prob)
    return filter_vmap(_qlaplace)(p, loc, scale)


@filter_jit
def _rlaplace(
    key: KeyArray,
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
    dtype=jnp.float_,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    loc = jnp.broadcast_to(loc, sample_shape)
    scale = jnp.broadcast_to(scale, sample_shape)
    return jrand.laplace(key, sample_shape, dtype=dtype) * scale + loc


def rlaplace(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Generates random Laplace samples from a given key.

    Args:
        key (KeyArray): The PRNG key to use for generating the samples.
        sample_shape (Optional[Shape], optional): The shape of the output array. Defaults to None.
        loc (Union[Float, ArrayLike], optional): The location parameter of the Laplace distribution. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter of the Laplace distribution. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to return the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: An array containing the random Laplace samples.

    Example:
        >>> rlaplace(key, (2, 3))
        Array([[-0.16134426,  1.6125823 , -0.6615164 ],
                [-1.5528525 , -0.21459664,  0.09816013]], dtype=float32)

    """
    rvs = _rlaplace(key, loc, scale, sample_shape, dtype=dtype)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
