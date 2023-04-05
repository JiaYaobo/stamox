import jax.numpy as jnp
from equinox import filter_jit, filter_vmap
from jaxtyping import ArrayLike, Bool
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
) -> ArrayLike:
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
) -> ArrayLike:
    q = jnp.atleast_1d(q)
    p = filter_vmap(_dbinom)(q, size, prob)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p

@filter_jit
def _qbinom(p, size, prob) -> ArrayLike:
    bino = tfp_Binomial(total_count=size, probs=prob)
    return bino.quantile(p)

@make_partial_pipe
def qbinom(
    p: ArrayLike,
    size: ArrayLike,
    prob: ArrayLike,
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> ArrayLike:
    p = jnp.atleast_1d(p)
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.exp(p)
    q = filter_vmap(_qbinom)(p, size, prob)
    return q

@filter_jit
def _rbinom(key, n, prob, size) -> ArrayLike:
    bino = tfp_Binomial(total_count=n, probs=prob)
    return bino.sample(sample_shape=size, seed=key)

@make_partial_pipe
def rbinom(
    key: ArrayLike,
    n: ArrayLike,
    prob: ArrayLike,
    size: ArrayLike,
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> ArrayLike:
    rvs = filter_vmap(_rbinom)(key, n, prob, size)
    if not lower_tail:
        rvs = 1 - rvs
    if log_prob:
        rvs = jnp.log(rvs)
    return rvs
