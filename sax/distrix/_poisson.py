import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import jit, vmap
from jax.scipy.special import gammainc


from sax.util import zero_dim_to_1_dim_array
from numpyro.distributions import Poisson


def ppoisson(x, rate):
    x = zero_dim_to_1_dim_array(x)
    p = vmap(_ppoisson, in_axes=(0, None))(x, rate)
    return p


@ft.partial(jit, static_argnames=('rate', ))
def _ppoisson(x, rate):
    k = jnp.floor(x) + 1
    return gammainc(k, rate)


def rpoisson(key, rate, sample_shape=()):
    return _rpoisson(key, rate, sample_shape)

def _rpoisson(key, rate, sample_shape=()):
    return jrand.poisson(key, rate, shape=sample_shape)
