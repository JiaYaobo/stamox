import functools as ft

import jax.random as jrand
from jax import jit, grad
from jax.scipy.special import betainc
from tensorflow_probability.substrates.jax.math import special as tfp_special

from ..maps import auto_map


def dbeta(x, a, b):
    _dnorm = grad(_pbeta)
    grads = auto_map(_dnorm, x, a, b)
    return grads


def pbeta(x, a, b):
    p = auto_map(_pbeta, x, a, b)
    return p


def qbeta(q, a, b):
    x = auto_map(_qbeta, q, a, b)
    return x


@jit
def _qbeta(q, a, b):
    return tfp_special.betaincinv(a, b, q)


@jit
def _pbeta(x, a, b):
    return betainc(a, b, x)


def rbeta(key, a, b, sample_shape=()):
    return _rbeta(key, a, b, sample_shape)


@ft.partial(jit, static_argnames=('a', 'b', 'sample_shape', ))
def _rbeta(key, a, b, sample_shape=()):
    return jrand.beta(key, a, b, sample_shape)
