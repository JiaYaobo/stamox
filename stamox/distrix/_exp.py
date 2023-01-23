import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import jit, grad

from .. maps import auto_map


def dexp(x, rate):
    _dexp = grad(_pexp)
    grads = auto_map(_dexp,x, rate)
    return grads


def pexp(x, rate):
    p = auto_map(_pexp, x, rate)
    return p


def qexp(q, rate):
    x = auto_map(_qexp, q, rate)
    return x


@jit
def _pexp(x, rate):
    return -jnp.expm1(-rate * x)


@jit
def _qexp(q, rate):
    return -jnp.log1p(-q) / rate


def rexp(key, rate, sample_shape=()):
    return _rexp(key, sample_shape=sample_shape) / rate


@ft.partial(jit, static_argnames=('sample_shape', ))
def _rexp(key, sample_shape=()):
    return jrand.exponential(key, shape=sample_shape)
