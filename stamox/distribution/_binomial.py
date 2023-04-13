from typing import Optional

import jax.numpy as jnp
import numpy as np
from equinox import filter_jit, filter_vmap
from jax import pure_callback, ShapeDtypeStruct
from jax._src.random import KeyArray, Shape
from jaxtyping import ArrayLike, Bool
from scipy.stats import binom
from tensorflow_probability.substrates.jax.distributions import Binomial as tfp_Binomial

from ..core import make_partial_pipe


@filter_jit
def _pbinom(q, size, prob) -> ArrayLike:
    bino = tfp_Binomial(total_count=size, probs=prob)
    return bino.cdf(q)


@make_partial_pipe
def pbinom(
    q: ArrayLike,
    size: ArrayLike,
    prob: ArrayLike,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float32,
) -> ArrayLike:
    """Calculates the cumulative probability of a binomial distribution.

    Args:
        q (ArrayLike): The quantiles to compute.
        size (ArrayLike): The number of trials.
        prob (ArrayLike): The probability of success in each trial.
        lower_tail (Bool, optional): If True (default), the lower tail probability is returned.
        log_prob (Bool, optional): If True, the logarithm of the probability is returned.
        dtype (optional): The data type of the output array. Defaults to jnp.float32.

    Returns:
        ArrayLike: The cumulative probability of the binomial distribution.

    Example:
        >>> q = jnp.array([0.1, 0.5, 0.9])
        >>> size = 10
        >>> prob = 0.5
        >>> pbinom(q, size, prob)
    """
    q = jnp.asarray(q, dtype=dtype)
    q = jnp.atleast_1d(q)
    p = filter_vmap(_pbinom)(q, size, prob)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


@filter_jit
def _dbinom(q, size, prob) -> ArrayLike:
    bino = tfp_Binomial(total_count=size, probs=prob)
    return bino.prob(q)


@make_partial_pipe
def dbinom(
    q: ArrayLike,
    size: ArrayLike,
    prob: ArrayLike,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float32,
) -> ArrayLike:
    """Computes the probability of a binomial distribution.

    Args:
        q (ArrayLike): The value to compute the probability for.
        size (ArrayLike): The number of trials in the binomial distribution.
        prob (ArrayLike): The probability of success in each trial.
        lower_tail (Bool, optional): Whether to compute the lower tail probability. Defaults to True.
        log_prob (Bool, optional): Whether to return the logarithm of the probability. Defaults to False.
        dtype (jnp.float32, optional): The data type of the output array. Defaults to jnp.float32.

    Returns:
        ArrayLike: The probability of the binomial distribution.
    """
    q = jnp.asarray(q, dtype=dtype)
    q = jnp.atleast_1d(q)
    p = filter_vmap(_dbinom)(q, size, prob)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


@filter_jit
def _qbinom(p, size, prob, dtype) -> ArrayLike:
    result_shape_type = ShapeDtypeStruct(jnp.shape(p), dtype)
    _scp_binom_ppf = lambda x: binom(size, prob).ppf(x).astype(np.int32)
    q = pure_callback(_scp_binom_ppf, result_shape_type, p)
    return q


@make_partial_pipe
def qbinom(
    p: ArrayLike,
    size: ArrayLike,
    prob: ArrayLike,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.int32,
) -> ArrayLike:
    """Computes the quantile of a binomial distribution.

    Args:
        p (ArrayLike): The probability of success.
        size (ArrayLike): The number of trials.
        prob (ArrayLike): The probability of success in each trial.
        lower_tail (Bool, optional): Whether to compute the lower tail or not. Defaults to True.
        log_prob (Bool, optional): Whether to compute the log probability or not. Defaults to False.
        dtype (jnp.int32, optional): The data type of the output array. Defaults to jnp.int32.

    Returns:
        ArrayLike: The quantile of the binomial distribution.
    """
    p = jnp.asarray(p)
    p = jnp.atleast_1d(p)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.exp(p)

    q = filter_vmap(_qbinom)(p, size, prob, dtype)
    return q


@filter_jit
def _rbinom(key, n, prob, sample_shape, dtype) -> ArrayLike:
    bino = tfp_Binomial(total_count=n, probs=prob)
    return bino.sample(sample_shape=sample_shape, seed=key).astype(dtype)


@make_partial_pipe
def rbinom(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    n: ArrayLike = None,
    prob: ArrayLike = None,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float32,
) -> ArrayLike:
    """Generates random binomial samples from a given probability distribution.

    Args:
        key (KeyArray): A random number generator key.
        sample_shape (Optional[Shape], optional): The shape of the output array. Defaults to None.
        n (ArrayLike, optional): The number of trials. Defaults to None.
        prob (ArrayLike, optional): The probability of success for each trial. Defaults to None.
        lower_tail (Bool, optional): Whether to return the lower tail of the distribution. Defaults to True.
        log_prob (Bool, optional): Whether to return the logarithm of the probability. Defaults to False.
        dtype (jnp.float32, optional): The data type of the output array. Defaults to jnp.float32.

    Returns:
        ArrayLike: An array containing the random binomial samples.
    """
    rvs = _rbinom(key, n, prob, sample_shape, dtype)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
