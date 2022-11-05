import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import jit, vmap, grad
from jax.scipy.special import ndtr, ndtri


def dnorm(x, mean=0., sigma=1.):
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = jnp.expand_dims(x, axis=0)
    _dnorm = grad(_pnorm)
    grads = vmap(_dnorm, in_axes=(0, None, None))(x, mean, sigma)
    return grads


def pnorm(x, mean=0., sigma=1.):
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = jnp.expand_dims(x, axis=0)
    p = vmap(_pnorm, in_axes=(0, None, None))(x, mean, sigma)
    return p


@ft.partial(jit, static_argnames=("mean", "sigma",))
def _pnorm(x, mean=0., sigma=1.):
    scaled = (x - mean) / sigma
    return ndtr(scaled)


def qnorm(q, mean=0., sigma=1.):
    q = jnp.asarray(q)
    if q.ndim == 1:
        q = jnp.expand_dims(q, axis=0)
    q = vmap(_qnorm, in_axes=(0, None, None))(q, mean, sigma)
    return q


@ft.partial(jit, static_argnames=("mean", "sigma",))
def _qnorm(q, mean=0., sigma=1.):
    x = ndtri(q)
    return x * sigma + mean


def rnorm(key, mean=0., sigma=1., sample_shape=()):
    return _rnorm(key, sample_shape) * sigma + mean


def _rnorm(key, sample_shape=()):
    return jrand.normal(key, sample_shape)
