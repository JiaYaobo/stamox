import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit
from jax.scipy.special import betainc
from tensorflow_probability.substrates.jax.math import special as tfp_special


def pbeta(x, a, b):
    x = jnp.asarray(x)
    p = vmap(_pbeta, in_axes=(0, None, None))(x, a, b)
    return p

def qbeta(q, a, b):
    q = jnp.asarray(q)
    x = vmap(_qbeta, in_axes=(0, None, None))(q, a, b)
    return x

@ft.partial(jit, static_argnames=('a', 'b', ))
def _qbeta(q, a, b):
    return tfp_special.betaincinv(a, b, q)

@ft.partial(jit, static_argnames=('a', 'b', ))
def _pbeta(x, a, b):
    return betainc(a, b, x)


def rbeta(key, a, b, sample_shape=()):
    return _rbeta(key, a, b, sample_shape)


def _rbeta(key, a, b, sample_shape=()):
    return jrand.beta(key, a, b, sample_shape)
