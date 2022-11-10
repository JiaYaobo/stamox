import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import jit, vmap, grad

from stamox.math.special import fdtri, fdtr
from stamox.util import zero_dim_to_1_dim_array
from ._chisq import rchisq


def dF(x, dfn, dfd):
    x = jnp.asarray(x)
    x = zero_dim_to_1_dim_array(x)
    _df = grad(_pf)
    grads = vmap(_df, in_axes=(0, None, None))(x, dfn, dfd)
    return grads


def pF(x, dfn, dfd):
    x = jnp.asarray(x)
    x = zero_dim_to_1_dim_array(x)
    p = vmap(_pf, in_axes=(0, None, None))(x, dfn, dfd)
    return p


def qF(q, dfn, dfd):
    q = jnp.asarray(q)
    q = zero_dim_to_1_dim_array(q)
    x = vmap(_qf, in_axes=(0, None, None))(q, dfn, dfd)
    return x


@ft.partial(jit, static_argnames=('dfn', 'dfd', ))
def _pf(x, dfn, dfd):
    return fdtr(dfn, dfd, x)


@ft.partial(jit, static_argnames=('dfn', 'dfd', ))
def _qf(q, dfn, dfd):
    return fdtri(dfn, dfd, q)



def rF(key, dfn, dfd, sample_shape=()):
    k1, k2 = jrand.split(key)
    return (rchisq(k1, dfn, sample_shape=sample_shape)/dfn)/(rchisq(k2, dfd, sample_shape=sample_shape)/dfd)
