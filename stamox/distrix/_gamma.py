import functools as ft

import jax.random as jrand
from jax import jit, grad
from jax.scipy.special import gammainc
import tensorflow_probability.substrates.jax.math as tfp_math

from ..maps import auto_map


def dgamma(x, shape, rate):
    _dgamma = grad(_pgamma)
    grads = auto_map(_dgamma, x, shape, rate)
    return grads


def pgamma(x, shape=1., rate=1.):
    p = auto_map(_pgamma, x, shape, rate)
    return p


def qgamma(q, shape=1., rate=1.):
    x = auto_map(_qgamma, q, shape, rate)
    return x


@jit
def _qgamma(q, shape=1., rate=1.):
    return tfp_math.igammainv(shape, q) / rate


@jit
def _pgamma(x, shape=1., rate=1.):
    return gammainc(shape, x * rate)


def rgamma(key, shape=1., rate=1., sample_shape=()):
    return _rgamma(key, shape, sample_shape) / rate


@ft.partial(jit, static_argnames=('shape', 'sample_shape', ))
def _rgamma(key, shape, sample_shape=()):
    return jrand.gamma(key, shape, sample_shape)
