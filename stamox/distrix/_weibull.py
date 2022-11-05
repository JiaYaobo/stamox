import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit, grad


def dweibull(x, concentration=0., scale=1.):
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = jnp.expand_dims(x, axis=0)
    _dweibull = grad(_pweibull)
    grads = vmap(_dweibull, in_axes=(0, None, None))(x, concentration, scale)
    return grads


def pweibull(x,  concentration=0., scale=1.):
    x = jnp.asarray(x)
    p = vmap(_pweibull, in_axes=(0, None, None, None))(x, concentration, scale)
    return p


@ft.partial(jit, static_argnames=("concentration", "scale",))
def _pweibull(x, concentration=0., scale=1.):
    return 1 - jnp.exp(-((x / scale) ** concentration))


def qweibull(q, concentration=0., scale=1.):
    q = jnp.asarray(q)
    q = vmap(_qweibull, in_axes=(0, None, None))(q, concentration, scale)
    return q


@ft.partial(jit, static_argnames=("concentration", "scale",))
def _qweibull(q, concentration=0., scale=1.):
    x = jnp.float_power(-jnp.log(1 - q), 1/concentration) * scale
    return x


def rweibull(key, concentration=0., scale=1., sample_shape=()):
    return _rweibull(key, concentration, scale, sample_shape)


def _rweibull(key, concentration, scale, sample_shape=()):
    return jrand.weibull_min(key, scale, concentration, sample_shape)
