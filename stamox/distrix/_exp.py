import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit, grad

from stamox.util import zero_dim_to_1_dim_array


def dexp(x, rate):
    x = zero_dim_to_1_dim_array(x)
    _dexp = grad(_pexp)
    grads = vmap(_dexp, in_axes=(0, None))(x, rate)
    return grads

def pexp(x, rate):
    x = zero_dim_to_1_dim_array(x)
    p = vmap(_pexp, in_axes=(0, None))(x, rate)
    return p

def qexp(q, rate):
    q= zero_dim_to_1_dim_array(q)
    x = vmap(_qexp, in_axes=(0, None))(q, rate)
    return x

@ft.partial(jit, static_argnames=('rate', ))
def _pexp(x, rate):
    return -jnp.expm1(-rate * x)


@ft.partial(jit, static_argnames=('rate', ))
def _qexp(q, rate):
    return -jnp.log1p(-q) / rate

def rexp(key, rate, sample_shape=()):
    return _rexp(key, sample_shape=sample_shape) / rate

def _rexp(key, sample_shape=()):
    return jrand.exponential(key, shape=sample_shape)


    


