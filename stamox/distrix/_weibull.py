import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import jit, grad

from ..maps import auto_map


def dweibull(x, concentration=0., scale=1.):
    _dweibull = grad(_pweibull)
    grads = auto_map(_dweibull, x, concentration, scale)
    return grads


def pweibull(x,  concentration=0., scale=1.):
    p = auto_map(_pweibull, x, concentration, scale)
    return p


@jit
def _pweibull(x, concentration=0., scale=1.):
    return 1 - jnp.exp(-((x / scale) ** concentration))


def qweibull(q, concentration=0., scale=1.):
    x = auto_map(_qweibull, q, concentration, scale)
    return x


@jit
def _qweibull(q, concentration=0., scale=1.):
    x = jnp.float_power(-jnp.log(1 - q), 1/concentration) * scale
    return x


def rweibull(key, concentration=0., scale=1., sample_shape=()):
    return _rweibull(key, concentration, scale, sample_shape)

@ft.partial(jit, static_argnames=('concentration', 'scale', 'sample_shape'))
def _rweibull(key, concentration, scale, sample_shape=()):
    return jrand.weibull_min(key, scale, concentration, sample_shape)
