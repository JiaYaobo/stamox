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
def _pcauchy(
    x: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
) -> ArrayLike:
    dtype = lax.dtype(x)
    loc = jnp.asarray(loc, dtype=dtype)
    scale = jnp.asarray(scale, dtype=dtype)
    scaled = (x - loc) / scale
    return jnp.arctan(scaled) / jnp.pi + 0.5


def pcauchy(
    q: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculates the cumulative denisty probability c function of the Cauchy distribution.

    Args:
        q (Union[Float, ArrayLike]): The value at which to evaluate the CDF.
        loc (Union[Float, ArrayLike], optional): The location parameter of the Cauchy distribution. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter of the Cauchy distribution. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to return the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The cumulative density function of the Cauchy distribution.

    Example:
        >>> pcauchy(1.0, loc=0.0, scale=1.0, lower_tail=True, log_prob=False)
        Array([0.75], dtype=float32, weak_type=True)
    """
    q, dtype = _promote_dtype_to_floating(q, dtype)
    q = jnp.atleast_1d(q)
    q = _check_clip_distribution_domain(q)
    p = filter_vmap(_pcauchy)(q, loc, scale)
    p = _post_process(p, lower_tail, log_prob)
    return p


_dcauchy = filter_grad(filter_jit(_pcauchy))


def dcauchy(
    x: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the pdf of the Cauchy distribution.

    Args:
        x (Union[Float, ArrayLike]): The input values.
        loc (Union[Float, ArrayLike], optional): The location parameter. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to compute the lower tail. Defaults to True.
        log_prob (Bool, optional): Whether to compute the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The pdf of the Cauchy distribution.

    Example:
        >>> dcauchy(1.0, loc=0.0, scale=1.0)
        Array([0.15915494], dtype=float32, weak_type=True)
    """
    x, dtype = _promote_dtype_to_floating(x, dtype)
    x = jnp.atleast_1d(x)
    x = _check_clip_distribution_domain(x)
    grads = filter_vmap(_dcauchy)(x, loc, scale)
    grads = _post_process(grads, lower_tail, log_prob)
    return grads


@filter_jit
def _qcauchy(
    q: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
):
    dtype = lax.dtype(q)
    loc = jnp.asarray(loc, dtype=dtype)
    scale = jnp.asarray(scale, dtype=dtype)
    return lax.add(loc, lax.mul(scale, lax.tan(lax.mul(jnp.pi, lax.sub(q, 0.5)))))


def qcauchy(
    q: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the quantile of the Cauchy distribution.

    Args:
        q (Union[float, array-like]): Quantiles to compute.
        loc (Union[float, array-like], optional): Location parameter. Defaults to 0.0.
        scale (Union[float, array-like], optional): Scale parameter. Defaults to 1.0.
        lower_tail (bool, optional): Whether to compute the lower tail. Defaults to True.
        log_prob (bool, optional): Whether to compute the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The quantiles of the Cauchy distribution.

    Example:
        >>> qcauchy(0.5, loc=1.0, scale=2.0)
        Array([1.], dtype=float32, weak_type=True)
    """
    q, dtype = _promote_dtype_to_floating(q, dtype)
    q = jnp.atleast_1d(q)
    q = _check_clip_probability(q, lower_tail, log_prob)
    return filter_vmap(_qcauchy)(q, loc, scale)


@filter_jit
def _rcauchy(
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
    loc = jnp.asarray(loc, dtype=dtype)
    scale = jnp.asarray(scale, dtype=dtype)
    return jrand.cauchy(key, sample_shape, dtype=dtype) * scale + loc


def rcauchy(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Generates random samples from the Cauchy distribution.

    Args:
        key: A PRNGKey to use for generating the samples.
        sample_shape: The shape of the output array.
        loc: The location parameter of the Cauchy distribution.
        scale: The scale parameter of the Cauchy distribution.
        lower_tail: Whether to return the lower tail probability.
        log_prob: Whether to return the log probability.
        dtype: The dtype of the output.

    Returns:
        ArrayLike: An array of samples from the Cauchy distribution.

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> rcauchy(key, sample_shape=(2, 3), loc=0.0, scale=1.0)
        Array([[ 0.23841971, -3.0880406 ,  0.9507532 ],
                [ 2.8963416 ,  0.31303588, -0.14792857]], dtype=float32)
    """
    rvs = _rcauchy(key, loc, scale, sample_shape, dtype)
    rvs = _post_process(rvs, lower_tail, log_prob)
    return rvs
