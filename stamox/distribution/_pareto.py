from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax._src.random import Shape
from jax.random import KeyArray
from jaxtyping import ArrayLike, Bool, Float

from ._utils import (
    _check_clip_distribution_domain,
    _check_clip_probability,
    _post_process,
    _promote_dtype_to_floating,
    svmap_,
)


@filter_jit
def _ppareto(
    x: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
):
    return 1 - jnp.power(scale / x, alpha)


def ppareto(
    q: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    lower_tail: bool = True,
    log_prob: bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the cumulative distribution function of the Pareto distribution.

    Args:
        q (Union[Float, ArrayLike]): The value at which to evaluate the CDF.
        scale (Union[Float, ArrayLike]): The scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): The shape parameter of the Pareto distribution.
        lower_tail (bool, optional): Whether to compute the lower tail of the CDF. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The cumulative distribution function of the Pareto distribution evaluated at `q`.

    Example:
        >>> ppareto(0.2, 0.1, 2.0)
        Array(0.75, dtype=float32, weak_type=True)
    """
    q, _ = _promote_dtype_to_floating(q, dtype)
    q = svmap_(_check_clip_distribution_domain, q, scale)
    p = svmap_(_ppareto, q, scale, alpha)
    p = _post_process(p, lower_tail, log_prob)
    return p


_dpareto = filter_grad(filter_jit(_ppareto))


def dpareto(
    x: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the density of the Pareto distribution.

    Args:
        x (Union[Float, ArrayLike]): The value at which to evaluate the density.
        scale (Union[Float, ArrayLike]): The scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): The shape parameter of the Pareto distribution.
        lower_tail (bool, optional): Whether to compute the lower tail probability. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The density of the Pareto distribution evaluated at `x`.

    Example:
        >>> dpareto(0.2, 0.1, 2.0)
        Array([2.4999998], dtype=float32, weak_type=True)
    """
    x, _ = _promote_dtype_to_floating(x, dtype)
    grads = svmap_(_dpareto, x, scale, alpha)
    grads = _post_process(grads, lower_tail, log_prob)
    return grads


@filter_jit
def _qpareto(
    q: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
):
    return scale / jnp.power(1 - q, 1 / alpha)


def qpareto(
    p: Union[Float, ArrayLike],
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the quantile function of the Pareto distribution.

    Args:
        p (Union[Float, ArrayLike]): Quantiles to compute.
        scale (Union[Float, ArrayLike]): Scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): Shape parameter of the Pareto distribution.
        lower_tail (Bool, optional): Whether to compute the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to compute the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The quantiles of the Pareto distribution.

    Example:
        >>> qpareto(0.2, 0.1, 2.0)
        Array([0.1118034], dtype=float32, weak_type=True)
    """
    p, _ = _promote_dtype_to_floating(p, dtype)
    p = _check_clip_probability(p, lower_tail, log_prob)
    return svmap_(_qpareto, p, scale, alpha)


@filter_jit
def _rpareto(
    key: KeyArray,
    scale: Union[Float, ArrayLike],
    alpha: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
    dtype=jnp.float_,
):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(scale), jnp.shape(alpha))
    scale = jnp.broadcast_to(scale, sample_shape)
    alpha = jnp.broadcast_to(alpha, sample_shape)
    return jrand.pareto(key, alpha, shape=sample_shape, dtype=dtype) * scale


def rpareto(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    scale: Union[Float, ArrayLike] = None,
    alpha: Union[Float, ArrayLike] = None,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Generate random variable following a Pareto distribution.

    Args:
        key (KeyArray): A random number generator key.
        sample_shape (Optional[Shape], optional): The shape of the samples to be drawn. Defaults to None.
        scale (Union[Float, ArrayLike]): The scale parameter of the Pareto distribution.
        alpha (Union[Float, ArrayLike]): The shape parameter of the Pareto distribution.
        lower_tail (Bool, optional): Whether to calculate the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: random variable following a Pareto distribution.

    Example:
        >>> rpareto(jax.random.PRNGKey(0), sample_shape=(2, 3), scale=0.1, alpha=2.0)
        Array([[0.15330292, 0.10539087, 0.19686179],
                [0.30740616, 0.15743963, 0.13524036]], dtype=float32)
    """
    rvs = _rpareto(key, scale, alpha, sample_shape, dtype=dtype)
    rvs = _post_process(rvs, lower_tail, log_prob)
    return rvs
