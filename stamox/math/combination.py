import jax.numpy as jnp
from equinox import filter_jit, filter_vmap
from jax import lax
from jax.scipy.special import gammaln

from ..core import make_partial_pipe


@make_partial_pipe
def combination(k, n, dtype=jnp.int32):
    """Calculates the number of combinations of k elements from a set of n elements.

    Args:
        k (int or array): Number of elements to choose from the set.
        n (int): Size of the set.

    Returns:
        int or array: The number of combinations of k elements from a set of n elements.

    Example:
        >>> combination(2, 5)
        10
    """
    k = jnp.asarray(k, dtype=dtype)
    k = jnp.atleast_1d(k)
    return filter_vmap(_comb)(k, n)


@filter_jit
def _cal_comb(k, n):
    log_kfrac = gammaln(k + 1)
    log_nfrac = gammaln(n + 1)
    log_n_kfrac = gammaln(n - k + 1)
    comb = jnp.exp(log_nfrac - log_kfrac - log_n_kfrac)
    comb = jnp.round(comb)
    comb = jnp.asarray(comb, dtype=jnp.int32)
    return comb


@filter_jit
def _comb(k, n):
    if_illegal = jnp.where(jnp.logical_or(k > n, k < 0), 1, 0)

    def func1(nn, kk):
        return jnp.asarray(0, dtype=jnp.int32)

    def func2(nn, kk):
        return _cal_comb(nn, kk)

    return jnp.asarray(lax.cond(if_illegal, func1, func2, k, n))
