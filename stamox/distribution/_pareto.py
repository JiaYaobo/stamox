from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax._src.random import Shape
from jax.random import KeyArray
from jaxtyping import ArrayLike, Bool, Float

from ..core import make_partial_pipe


@filter_jit
def _ppareto(
    x: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
):
    return 1 - jnp.power(scale / x, alpha)


@make_partial_pipe
def ppareto(
    q: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    lower_tail: bool = True,
    log_prob: bool = False,
) -> ArrayLike:
    """Computes the cumulative distribution function of the Pareto distribution.

    Args:
        q (Union[Float, ArrayLike]): The value at which to evaluate the CDF.
        scale (Union[Float, ArrayLike]): The scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): The shape parameter of the Pareto distribution.
        lower_tail (bool, optional): Whether to compute the lower tail of the CDF. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        ArrayLike: The cumulative distribution function of the Pareto distribution evaluated at `q`.

    Example:
        >>> ppareto(0.2, 0.1, 2.0)
        Array([0.75], dtype=float32, weak_type=True)
    """
    q = jnp.asarray(q)
    q = jnp.atleast_1d(q)
    p = filter_vmap(_ppareto)(q, scale, alpha)
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
) -> ArrayLike:
    """Computes the density of the Pareto distribution.

    Args:
        x (Union[Float, ArrayLike]): The value at which to evaluate the density.
        scale (Union[Float, ArrayLike]): The scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): The shape parameter of the Pareto distribution.
        lower_tail (bool, optional): Whether to compute the lower tail probability. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        ArrayLike: The density of the Pareto distribution evaluated at `x`.

    Example:
        >>> dpareto(0.2, 0.1, 2.0)
        Array([2.4999998], dtype=float32, weak_type=True)
    """
    x = jnp.asarray(x)
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
):
    return scale / jnp.power(1 - q, 1 / alpha)


@make_partial_pipe
def qpareto(
    p: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> ArrayLike:
    """Computes the quantile function of the Pareto distribution.

    Args:
        p (Union[Float, ArrayLike]): Quantiles to compute.
        scale (Union[Float, ArrayLike]): Scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): Shape parameter of the Pareto distribution.
        lower_tail (Bool, optional): Whether to compute the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to compute the log probability. Defaults to False.

    Returns:
        ArrayLike: The quantiles of the Pareto distribution.

    Example:
        >>> qpareto(0.2, 0.1, 2.0)
        Array([0.1118034], dtype=float32, weak_type=True)
    """
    p = jnp.asarray(p)
    p = jnp.atleast_1d(p)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.exp(p)
    return filter_vmap(_qpareto)(p, scale, alpha)


@filter_jit
def _rpareto(
    key: KeyArray,
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(scale), jnp.shape(alpha))
    scale = jnp.broadcast_to(scale, sample_shape)
    alpha = jnp.broadcast_to(alpha, sample_shape)
    return jrand.pareto(key, alpha, shape=sample_shape) * scale


@make_partial_pipe
def rpareto(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    scale: Union[Float, ArrayLike] = None,
    alpha: Union[Float, ArrayLike] = None,
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> ArrayLike:
    """Generate random variable following a Pareto distribution.

    Args:
        key (KeyArray): A random number generator key.
        sample_shape (Optional[Shape], optional): The shape of the samples to be drawn. Defaults to None.
        scale (Union[Float, ArrayLike]): The scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): The shape parameter of the Pareto distribution.
        lower_tail (Bool, optional): Whether to calculate the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to return the log probability. Defaults to False.

    Returns:
        ArrayLike: random variable following a Pareto distribution.

    Example:
        >>> rpareto(jax.random.PRNGKey(0), sample_shape=(2, 3), scale=0.1, alpha=2.0)
        Array([[0.15330292, 0.10539087, 0.19686179],
                [0.30740616, 0.15743963, 0.13524036]], dtype=float32)
    """
    rvs = _rpareto(key, scale, alpha, sample_shape)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
