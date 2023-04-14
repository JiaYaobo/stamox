from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax._src.random import Shape
from jax.random import KeyArray
from jaxtyping import ArrayLike, Bool, Float

from ..core import make_partial_pipe


@filter_jit
def _pcauchy(
    x: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
) -> ArrayLike:
    scaled = (x - loc) / scale
    return jnp.arctan(scaled) / jnp.pi + 0.5


@make_partial_pipe
def pcauchy(
    q: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype = jnp.float32,
) -> ArrayLike:
    """Calculates the cumulative denisty probability c function of the Cauchy distribution.

    Args:
        q (Union[Float, ArrayLike]): The value at which to evaluate the CDF.
        loc (Union[Float, ArrayLike], optional): The location parameter of the Cauchy distribution. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter of the Cauchy distribution. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to return the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float32.

    Returns:
        ArrayLike: The cumulative density function of the Cauchy distribution.

    Example:
        >>> pcauchy(1.0, loc=0.0, scale=1.0, lower_tail=True, log_prob=False)
        Array([0.75], dtype=float32, weak_type=True)
    """
    q = jnp.asarray(q, dtype=dtype)
    q = jnp.atleast_1d(q)
    p = filter_vmap(_pcauchy)(q, loc, scale)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


_dcauchy = filter_grad(filter_jit(_pcauchy))


@make_partial_pipe
def dcauchy(
    x: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype = jnp.float32,
) -> ArrayLike:
    """Computes the pdf of the Cauchy distribution.

    Args:
        x (Union[Float, ArrayLike]): The input values.
        loc (Union[Float, ArrayLike], optional): The location parameter. Defaults to 0.0.
        scale (Union[Float, ArrayLike], optional): The scale parameter. Defaults to 1.0.
        lower_tail (Bool, optional): Whether to compute the lower tail. Defaults to True.
        log_prob (Bool, optional): Whether to compute the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float32.

    Returns:
        ArrayLike: The pdf of the Cauchy distribution.

    Example:
        >>> dcauchy(1.0, loc=0.0, scale=1.0, lower_tail=True, log_prob=False)
        Array([0.15915494], dtype=float32, weak_type=True)
    """
    x = jnp.asarray(x, dtype=dtype)
    x = jnp.atleast_1d(x)
    grads = filter_vmap(_dcauchy)(x, loc, scale)
    if not lower_tail:
        grads = 1 - grads
    if log_prob:
        grads = jnp.log(grads)
    return grads


@filter_jit
def _qcauchy(
    q: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
):
    return loc + scale * jnp.tan(jnp.pi * (q - 0.5))


@make_partial_pipe
def qcauchy(
    q: Union[Float, ArrayLike],
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype = jnp.float32,
) -> ArrayLike:
    """Computes the quantile of the Cauchy distribution.

    Args:
        q (Union[float, array-like]): Quantiles to compute.
        loc (Union[float, array-like], optional): Location parameter. Defaults to 0.0.
        scale (Union[float, array-like], optional): Scale parameter. Defaults to 1.0.
        lower_tail (bool, optional): Whether to compute the lower tail. Defaults to True.
        log_prob (bool, optional): Whether to compute the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float32.

    Returns:
        ArrayLike: The quantiles of the Cauchy distribution.

    Example:
        >>> qcauchy(0.5, loc=1.0, scale=2.0, lower_tail=True, log_prob=False)
        Array([1.], dtype=float32, weak_type=True)
    """
    q = jnp.asarray(q, dtype=dtype)
    q = jnp.atleast_1d(q)
    if not lower_tail:
        q = 1 - q
    if log_prob:
        q = jnp.exp(q)
    return filter_vmap(_qcauchy)(q, loc, scale)


def _rcauchy(
    key: KeyArray,
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
    dtype = jnp.float32,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    loc = jnp.broadcast_to(loc, sample_shape)
    scale = jnp.broadcast_to(scale, sample_shape)
    loc = jnp.asarray(loc, dtype=dtype)
    scale = jnp.asarray(scale, dtype=dtype)
    return jrand.cauchy(key, sample_shape, dtype=dtype) * scale + loc


@make_partial_pipe
def rcauchy(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    loc: Union[Float, ArrayLike] = 0.0,
    scale: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype = jnp.float32,
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
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
