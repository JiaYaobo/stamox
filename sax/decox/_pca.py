from math import log, sqrt
import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jrand
import tensorflow_probability.substrates.jax.distributions as jd
from jax import lax, jit, vmap


def _pca(x, n_components):
    n_samples, n_features = x.shape

    pass
