from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax._src.random import Shape
from jax.random import KeyArray
from jaxtyping import Array, ArrayLike, Bool, Float

from ..core import make_partial_pipe


@filter_jit
def _ppareto(
    x: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
) -> Array:
    return 1 - jnp.power(scale / x, alpha)


@make_partial_pipe
def ppareto(
    x: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
) -> Array:
    """Computes the cumulative distribution function of the Pareto distribution.

    Args:
        x (Union[Float, ArrayLike]): The value at which to evaluate the CDF.
        scale (Union[Float, ArrayLike]): The scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): The shape parameter of the Pareto distribution.
        lower_tail (bool, optional): Whether to compute the lower tail of the CDF. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        Array: The cumulative distribution function of the Pareto distribution evaluated at `x`.
    """
    x = jnp.atleast_1d(x)
    p = filter_vmap(_ppareto)(x, scale, alpha)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


_dpareto = filter_grad(filter_jit(_ppareto))


@make_partial_pipe
def dpareto(
    x: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
) -> Array:
    """Computes the density of the Pareto distribution.

    Args:
        x (Union[Float, ArrayLike]): The value at which to evaluate the density.
        scale (Union[Float, ArrayLike]): The scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): The shape parameter of the Pareto distribution.
        lower_tail (bool, optional): Whether to compute the lower tail probability. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        Array: The density of the Pareto distribution evaluated at `x`.
    """
    x = jnp.atleast_1d(x)
    grads = filter_vmap(_dpareto)(x, scale, alpha)
    if not lower_tail:
        grads = 1 - grads
    if log_prob:
        grads = jnp.log(grads)
    return grads


@filter_jit
def _qpareto(
    q: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
) -> Array:
    return scale / jnp.power(1 - q, 1 / alpha)


@make_partial_pipe
def qpareto(
    q: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> Array:
    """Computes the quantile function of the Pareto distribution.

    Args:
        q (Union[Float, ArrayLike]): Quantiles to compute.
        scale (Union[Float, ArrayLike]): Scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): Shape parameter of the Pareto distribution.
        lower_tail (Bool, optional): Whether to compute the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to compute the log probability. Defaults to False.

    Returns:
        Array: The quantiles of the Pareto distribution.
    """

    q = jnp.atleast_1d(q)
    if not lower_tail:
        q = 1 - q
    if log_prob:
        q = jnp.exp(q)
    return filter_vmap(_qpareto)(q, scale, alpha)


@filter_jit
def _rpareto(
    key: KeyArray,
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
) -> Array:
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(scale), jnp.shape(alpha))
    scale = jnp.broadcast_to(scale, sample_shape)
    alpha = jnp.broadcast_to(alpha, sample_shape)
    return jrand.pareto(key, alpha, shape=sample_shape) * scale


@make_partial_pipe
def rpareto(
    key: KeyArray,
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> Array:
    """Generate random variable following a Pareto distribution.

    Args:
        key (KeyArray): A random number generator key.
        scale (Union[Float, ArrayLike]): The scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): The shape parameter of the Pareto distribution.
        sample_shape (Optional[Shape], optional): The shape of the samples to be drawn. Defaults to None.
        lower_tail (Bool, optional): Whether to calculate the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        Array: random variable following a Pareto distribution.
    """
    probs = _rpareto(key, scale, alpha, sample_shape)
    if not lower_tail:
        probs = 1 - probs
    if log_prob:
        probs = jnp.log(probs)
    return probs
