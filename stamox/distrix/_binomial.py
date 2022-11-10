import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit, grad

from ._bernoulli import rbernoulli


def rbinomial(key, p, n, sample_shape=()):
    return _rbinomial(key, p, n, sample_shape)

def _rbinomial(key, p, n, sample_shape=()):
    keys = jrand.split(key, n)
    rbins = vmap(rbernoulli, in_axes=(0, None, None))(keys, p, sample_shape)


