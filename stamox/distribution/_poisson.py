from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_jit, filter_vmap
from jax import pure_callback, ShapeDtypeStruct
from jax._src.random import Shape
from jax.random import KeyArray
from jax.scipy.special import gammainc, gammaln
from jaxtyping import Array, ArrayLike, Bool, Float
from scipy.stats import poisson

from ._utils import (
    _check_clip_distribution_domain,
    _check_clip_probability,
    _post_process,
    _promote_dtype_to_floating,
)


@filter_jit
def _ppoisson(x: Union[Float, ArrayLike], rate: Union[Float, ArrayLike]) -> Array:
    k = jnp.floor(x) + 1.0
    return 1 - gammainc(k, rate)


def ppoisson(
    q: Union[Float, ArrayLike],
    rate: Union[Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the cumulative distribution function of the Poisson distribution.

    Args:
        q (Union[Float, ArrayLike]): The value at which to evaluate the CDF.
        rate (Union[Float, ArrayLike]): The rate parameter of the Poisson distribution.
        lower_tail (bool, optional): Whether to compute the lower tail of the CDF. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The cumulative distribution function of the Poisson distribution evaluated at `q`.
    """
    q, dtype = _promote_dtype_to_floating(q, dtype)
    q = jnp.atleast_1d(q)
    q = _check_clip_distribution_domain(q, lower=0.0)
    p = filter_vmap(_ppoisson)(q, rate)
    p = _post_process(p, lower_tail=lower_tail, log_prob=log_prob)
    return p


@filter_jit
def _dpoisson(x, rate):
    e = jnp.exp(-rate)
    numerator = rate**x
    denominator = jnp.exp(gammaln(x + 1))
    return (numerator / denominator) * e


def dpoisson(
    x: Union[Float, ArrayLike],
    rate: Union[Float, ArrayLike],
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the probability density function of the Poisson distribution.

    Args:
        x (Union[Float, ArrayLike]): The value at which to evaluate the PDF.
        rate (Union[Float, ArrayLike]): The rate parameter of the Poisson distribution.
        lower_tail (bool, optional): Whether to compute the lower tail of the PDF. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The probability density function of the Poisson distribution evaluated at `x`.
    """
    x, _ = _promote_dtype_to_floating(x, dtype)
    x = jnp.atleast_1d(x)
    x = _check_clip_distribution_domain(x, lower=0.0)
    grads = filter_vmap(_dpoisson)(x, rate)
    grads = _post_process(grads, lower_tail=lower_tail, log_prob=log_prob)
    return grads


@filter_jit
def _rpoisson(
    key: KeyArray,
    rate: Union[Float, ArrayLike],
    sample_shape: Optional[Shape] = None,
    dtype=jnp.int_,
):
    return jrand.poisson(key, rate, shape=sample_shape, dtype=dtype)


def rpoisson(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    rate: Union[Float, ArrayLike] = None,
    lower_tail=True,
    log_prob=False,
    dtype=jnp.int_,
) -> ArrayLike:
    """Generates samples from the Poisson distribution.

    Args:
        key (KeyArray): Random number generator state used for random sampling.
        rate (Union[Float, ArrayLike]): The rate parameter of the Poisson distribution.
        sample_shape (Optional[Shape], optional): Shape of the output array. Defaults to None.
        lower_tail (bool, optional): Whether to return the lower tail probability. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.int_.

    Returns:
        ArrayLike: Samples from the Poisson distribution.
    """
    rvs = _rpoisson(key, rate, sample_shape=sample_shape, dtype=dtype)
    rvs = _post_process(rvs, lower_tail=lower_tail, log_prob=log_prob)
    return rvs


@filter_jit
def _qpoisson(q, rate, dtype):
    result_shape_type = ShapeDtypeStruct(jnp.shape(q), dtype)
    _scp_poisson_ppf = lambda q, rate: poisson(rate).ppf(q).astype(dtype)
    p = pure_callback(_scp_poisson_ppf, result_shape_type, q, rate)
    return p


def qpoisson(
    p: Union[Float, ArrayLike],
    rate: Union[Float, ArrayLike],
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.int_,
) -> ArrayLike:
    """Computes the quantile function of the Poisson distribution.

    Args:
        p (Union[Float, ArrayLike]): The probability at which to evaluate the quantile function.
        rate (Union[Float, ArrayLike]): The rate parameter of the Poisson distribution.
        lower_tail (bool, optional): Whether to compute the lower tail of the quantile function. Defaults to True.
        log_prob (bool, optional): Whether to return the log probability. Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.int_.

    Returns:
        ArrayLike: The quantile function of the Poisson distribution evaluated at `p`.
    """
    p, _ = _promote_dtype_to_floating(p, None)
    p = jnp.atleast_1d(p)
    p = _check_clip_probability(p, lower_tail, log_prob)
    q = filter_vmap(_qpoisson)(p, rate, dtype)
    return q
