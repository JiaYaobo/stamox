import jax.numpy as jnp

from jax import lax
from jax.scipy.special import gammaln
from equinox import filter_jit, filter_vmap

from ..core import make_partial_pipe


@make_partial_pipe
def choose(k, n):
    k = jnp.atleast_1d(k)
    return filter_vmap(_choose)(k, n)


@filter_jit
def _cal_choose(k, n):
    log_kfrac = gammaln(k + 1)
    log_nfrac = gammaln(n + 1)
    log_n_kfrac = gammaln(n - k + 1)
    comb = jnp.exp(log_nfrac - log_kfrac - log_n_kfrac)
    comb = jnp.round(comb)
    comb = jnp.asarray(comb, dtype=jnp.int32)
    return comb


@filter_jit
def _choose(k, n):
    if_illegal = jnp.where(jnp.logical_or(k > n, k < 0), 1, 0)
    def func1(nn, kk): return jnp.asarray(0, dtype=jnp.int32)
    def func2(nn, kk): return _cal_choose(nn, kk)
    return jnp.asarray(lax.cond(if_illegal, func1, func2, k, n))
