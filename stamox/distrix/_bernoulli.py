import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit, grad



def rbernoulli(key, p, sample_shape=()):
    return _rbernoulli(key, p, sample_shape)

def _rbernoulli(key, p, sample_shape=()):
    return jrand.bernoulli(key, p, shape=sample_shape)