import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit
from jax.scipy.special import gammainc
import  tensorflow_probability.substrates.jax.math as tfp_math


def pgamma(x, shape=1., rate=1.):
    x = jnp.asarray(x)
    p = vmap(_pgamma, in_axes=(0, None, None))(x, shape, rate)
    return p

def qgamma(q, shape=1., rate=1.):
    q = jnp.asarray(q)
    x = vmap(_qgamma, in_axes=(0, None, None))(q, shape, rate)
    return x

@ft.partial(jit, static_argnames=('shape','rate', ))
def _qgamma(q, shape=1., rate=1.): 
    return tfp_math.igammaincinv(shape / rate, q) / rate

@ft.partial(jit, static_argnames=('shape','rate', ))
def _pgamma(x, shape=1., rate=1.):
    return gammainc(shape / rate, x) * rate


def rgamma(key, shape=1., rate=1., sample_shape=()):
    return _rgamma(key, shape, rate, sample_shape)


def _rgamma(key, shape, rate, sample_shape=()):
    return jrand.gamma(key, shape / rate, sample_shape)