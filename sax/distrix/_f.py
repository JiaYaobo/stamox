import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import jit, vmap, grad

from sax.math.special import fdtri, fdtr


def dF(x, dfn, dfd):
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = jnp.expand_dims(x, axis=0)
    _df = grad(_pf)
    grads = vmap(_df, in_axes=(0, None, None))(x, dfn, dfd)
    return grads

def pF(x, dfn, dfd):
    if x.ndim == 1:
        x = jnp.expand_dims(x, axis=0)
    p = vmap(_pf, in_axes=(0, None, None))(x, dfn ,dfd)
    return p

def qF(q, dfn, dfd):
    if q.ndim == 1:
        q = jnp.expand_dims(q, axis=0)
    x = vmap(_qf, in_axes=(0, None, None))(q, dfn ,dfd)
    return x

@ft.partial(jit,static_argnames=('dfn', 'dfd', ))
def _pf(x, dfn, dfd):
    return fdtr(dfn, dfd, x)

@ft.partial(jit,static_argnames=('dfn', 'dfd', ))
def _qf(q, dfn, dfd):
    return fdtri(dfn, dfd, q)
