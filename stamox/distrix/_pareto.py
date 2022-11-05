import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit, grad

def dpareto(x, scale, alpha):
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = jnp.expand_dims(x, axis=0)
    _dpareto= grad(_ppareto)
    grads = vmap(_dpareto, in_axes=(0, None, None))(x, scale, alpha)
    return grads

def rpareto(key, scale, alpha, sample_shape=()):
    return _rpareto(key, scale, alpha, sample_shape)

def _rpareto(key, scale, alpha, sample_shape=()):
    return jrand.exponential(key, shape=sample_shape) / alpha * scale

def ppareto(x, scale, alpha):
    x = jnp.asarray(x)
    if x.ndim == 1:
        x = jnp.expand_dims(x, axis=0)
    
    p = vmap(_ppareto, in_axes=(0, None, None))(x, scale, alpha)
    return p

def qpareto(q, scale, alpha):
    q= jnp.asarray(q)
    if q.ndim == 1:
        q = jnp.expand_dims(q, axis=0)
    
    x = vmap(_qpareto, in_axes=(0, None, None))(q, scale, alpha)
    return x

@ft.partial(jit, static_argnames=('scale', 'alpha', ))
def _ppareto(x, scale, alpha):
    return 1 - jnp.power(scale / x, alpha)

@ft.partial(jit, static_argnames=('scale', 'alpha', ))
def _qpareto(q, scale, alpha):
    return scale / jnp.power(1 - q, 1 / alpha)