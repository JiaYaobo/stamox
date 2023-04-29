from typing import Optional

import jax.numpy as jnp
from equinox import filter_jit
from jax import pure_callback, ShapeDtypeStruct
from jax._src.random import KeyArray, Shape
from jaxtyping import ArrayLike, Bool
from scipy.stats import binom
from tensorflow_probability.substrates.jax.distributions import Binomial as tfp_Binomial

from ._utils import (
    _check_clip_distribution_domain,
    _check_clip_probability,
    _post_process,
    _promote_dtype_to_floating,
    svmap_,
)


@filter_jit
def _pbinom(q, size, prob) -> ArrayLike:
    size = jnp.asarray(size, dtype=q.dtype)
    prob = jnp.asarray(prob, dtype=q.dtype)
    bino = tfp_Binomial(total_count=size, probs=prob)
    return bino.cdf(q)


def pbinom(
    q: ArrayLike,
    size: ArrayLike,
    prob: ArrayLike,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculates the cumulative probability of a binomial distribution.

    Args:
        q (ArrayLike): The quantiles to compute.
        size (ArrayLike): The number of trials.
        prob (ArrayLike): The probability of success in each trial.
        lower_tail (Bool, optional): If True (default), the lower tail probability is returned.
        log_prob (Bool, optional): If True, the logarithm of the probability is returned.
        dtype (optional): The data type of the output array. Defaults to jnp.float_.

    Returns:
        ArrayLike: The cumulative probability of the binomial distribution.

    Example:
        >>> q = jnp.array([0.1, 0.5, 0.9])
        >>> size = 10
        >>> prob = 0.5
        >>> pbinom(q, size, prob)
    """
    q, dtype = _promote_dtype_to_floating(q, dtype)
    q = _check_clip_distribution_domain(q, 0, size)
    p = svmap_(_pbinom, q, size, prob)
    p = _post_process(p, lower_tail, log_prob)
    return p


@filter_jit
def _dbinom(q, size, prob) -> ArrayLike:
    size = jnp.asarray(size, dtype=q.dtype)
    prob = jnp.asarray(prob, dtype=q.dtype)
    bino = tfp_Binomial(total_count=size, probs=prob)
    return bino.prob(q)


def dbinom(
    q: ArrayLike,
    size: ArrayLike,
    prob: ArrayLike,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Computes the probability of a binomial distribution.

    Args:
        q (ArrayLike): The value to compute the probability for.
        size (ArrayLike): The number of trials in the binomial distribution.
        prob (ArrayLike): The probability of success in each trial.
        lower_tail (Bool, optional): Whether to compute the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to return the logarithm of the probability. Defaults to False.
        dtype (jnp.float_, optional): The data type of the output array. Defaults to jnp.float_.

    Returns:
        ArrayLike: The probability of the binomial distribution.

    Example:
        >>> q = jnp.array([0.1, 0.5, 0.9])
        >>> size = 10
        >>> prob = 0.5
        >>> dbinom(q, size, prob)
    """
    q, dtype = _promote_dtype_to_floating(q, dtype)
    d = svmap_(_dbinom, q, size, prob)
    d = _post_process(d, lower_tail, log_prob)
    return d


@filter_jit
def _qbinom(p, size, prob, dtype) -> ArrayLike:
    result_shape_type = ShapeDtypeStruct(jnp.shape(p), dtype)
    _scp_binom_ppf = lambda x: binom(size, prob).ppf(x).astype(dtype)
    q = pure_callback(_scp_binom_ppf, result_shape_type, p)
    return q


def qbinom(
    p: ArrayLike,
    size: ArrayLike,
    prob: ArrayLike,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.int_,
) -> ArrayLike:
    """Computes the quantile of a binomial distribution.

    Args:
        p (ArrayLike): The probability of success.
        size (ArrayLike): The number of trials.
        prob (ArrayLike): The probability of success in each trial.
        lower_tail (Bool, optional): Whether to compute the lower tail or not. Defaults to True.
        log_prob (Bool, optional): Whether to compute the log probability or not. Defaults to False.
        dtype (jnp.int_, optional): The data type of the output array. Defaults to jnp.int_.

    Returns:
        ArrayLike: The quantile of the binomial distribution.

    Example:
        >>> p = jnp.array([0.1, 0.5, 0.9])
        >>> size = 10
        >>> prob = 0.5
        >>> qbinom(p, size, prob)
    """
    if dtype is None:
        dtype = jnp.int_
    p = jnp.asarray(p, dtype=jnp.float_)
    p = _check_clip_probability(p, lower_tail, log_prob)
    q = _qbinom(p, size, prob, dtype)
    return q


@filter_jit
def _rbinom(key, size, prob, sample_shape, dtype) -> ArrayLike:
    bino = tfp_Binomial(total_count=size, probs=prob)
    return bino.sample(sample_shape=sample_shape, seed=key).astype(dtype)


def rbinom(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    size: ArrayLike = None,
    prob: ArrayLike = None,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.int_,
) -> ArrayLike:
    """Generates random binomial samples from a given probability distribution.

    Args:
        key (KeyArray): A random number generator key.
        sample_shape (Optional[Shape], optional): The shape of the output array. Defaults to None.
        size (ArrayLike, optional): The number of trials. Defaults to None.
        prob (ArrayLike, optional): The probability of success for each trial. Defaults to None.
        lower_tail (Bool, optional): Whether to return the lower tail of the distribution. Defaults to True.
        log_prob (Bool, optional): Whether to return the logarithm of the probability. Defaults to False.
        dtype (jnp.float32, optional): The data type of the output array. Defaults to jnp.float32.

    Returns:
        ArrayLike: An array containing the random binomial samples.

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> sample_shape = (3, 3)
        >>> size = 10
        >>> prob = 0.5
        >>> rbinom(key, sample_shape, size, prob)
    """
    rvs = _rbinom(key, size, prob, sample_shape, dtype)
    rvs = _post_process(rvs, lower_tail, log_prob)
    return rvs
