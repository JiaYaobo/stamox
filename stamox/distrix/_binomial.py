import functools as ft

import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
from jax import vmap, jit
from jax.scipy.special import gammaln

from ._bernoulli import rbernoulli
from stamox.util import zero_dim_to_1_dim_array


def dbinomial(k, n, p):
    k = jnp.asarray(k, dtype=jnp.int32)
    k = zero_dim_to_1_dim_array(k)
    n = int(n)
    p = float(p)
    dens = vmap(_dbinomial, in_axes=(0, None, None))(k, n, p)
    return dens


@ft.partial(jit, static_argnames=('n', 'p'))
def _dbinomial(k, n, p):
    k = jnp.array(k, jnp.int32)
    log_kfrac = gammaln(k + 1)
    log_nfrac = gammaln(n + 1)
    log_n_kfrac = gammaln(n-k + 1)
    comb = jnp.exp(log_nfrac - log_kfrac - log_n_kfrac)
    pp = jnp.power(p, k) * jnp.power(1-p, n-k) * comb
    return pp


def rbinomial(key, p, n, sample_shape=()):
    return _rbinomial(key, p, n, sample_shape)


def _rbinomial(key, p, n, sample_shape=()):
    keys = jrand.split(key, n)
    rbins = vmap(rbernoulli, in_axes=(0, None, None))(keys, p, sample_shape)
