import jax.numpy as jnp
import jax.random as jrand
from jax import jit
from jax.scipy.special import gammainc

from ..maps import auto_map


def ppoisson(x, rate):
    p = auto_map(_ppoisson, x, rate)
    return p

@jit
def _ppoisson(x, rate):
    k = jnp.floor(x) + 1.
    return gammainc(k, rate)


def rpoisson(key, rate, sample_shape=()):
    return _rpoisson(key, rate, sample_shape)

def _rpoisson(key, rate, sample_shape=()):
    return jrand.poisson(key, rate, shape=sample_shape)
