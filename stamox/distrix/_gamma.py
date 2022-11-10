import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit, grad
from jax.scipy.special import gammainc
import tensorflow_probability.substrates.jax.math as tfp_math

# from numpyro.distributions import Gamma


def dgamma(x, shape, rate):
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = jnp.expand_dims(x, axis=0)
    _dgamma = grad(_pgamma)
    grads = vmap(_dgamma, in_axes=(0, None, None))(x, shape, rate)
    return grads


def pgamma(x, shape=1., rate=1.):
    x = jnp.asarray(x)
    p = vmap(_pgamma, in_axes=(0, None, None))(x, shape, rate)
    return p


def qgamma(q, shape=1., rate=1.):
    q = jnp.asarray(q)
    x = vmap(_qgamma, in_axes=(0, None, None))(q, shape, rate)
    return x


@ft.partial(jit, static_argnames=('shape', 'rate', ))
def _qgamma(q, shape=1., rate=1.):
    return tfp_math.igammainv(shape, q) / rate


@ft.partial(jit, static_argnames=('shape', 'rate', ))
def _pgamma(x, shape=1., rate=1.):
    return gammainc(shape, x * rate) 


def rgamma(key, shape=1., rate=1., sample_shape=()):
    return _rgamma(key, shape, sample_shape) / rate


def _rgamma(key, shape, sample_shape=()):
    return jrand.gamma(key, shape, sample_shape)
