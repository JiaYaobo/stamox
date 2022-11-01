import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import jit, vmap


def punif(x, mini=0., maxi=1.):
    x = jnp.asarray(x)
    p = vmap(_punif, in_axes=(0, None, None))(x, mini, maxi)
    return p


@ft.partial(jit, static_argnames=("mini", "maxi",))
def _punif(x, mini=0., maxi=1.):
    p = (x - mini) / (maxi - mini)
    return p


def qunif(q,  mini=0., maxi=1.):
    q = jnp.asarray(q)
    q = vmap(_qunif, in_axes=(0, None, None))(q, mini, maxi)
    return q


@ft.partial(jit, static_argnames=("mini", "maxi",))
def _qunif(q, mini=0., maxi=1.):
    x = q * (maxi - mini) + mini
    return x


def runif(key, mini=0., maxi=1., sample_shape=()):
    return _runif(key, mini, maxi, sample_shape)


def _runif(key, mini, maxi, sample_shape=()):

    return jrand.uniform(key, sample_shape, minval=mini, maxval=maxi)