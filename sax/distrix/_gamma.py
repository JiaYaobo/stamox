import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit
from jax.scipy.special import gammainc
import  tensorflow_probability.substrates.jax.math as tfp_math


def pgamma(x, a):
    x = jnp.asarray(x)
    p = vmap(_pgamma, in_axes=(0, None))(x, a)
    return p

def qgamma(q, a):
    q = jnp.asarray(q)
    x = vmap(_qgamma, in_axes=(0, None))(q, a)
    return x

@ft.partial(jit, static_argnames=('a', ))
def _qgamma(q, a):
    return tfp_math.igammaincinv(a, q)

@ft.partial(jit, static_argnames=('a', ))
def _pgamma(x, a):
    return gammainc(a, x)


def rgamma(key, a, b, sample_shape=()):
    return _rgamma(key, a, b, sample_shape)


def _rgamma(key, a, sample_shape=()):
    return jrand.gamma(key, a, sample_shape)