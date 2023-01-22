import jax.numpy as jnp
import jax.random as jrand
from jax import jit, vmap, grad
from jax.scipy.special import ndtr, ndtri

from ..maps import auto_map


def dnorm(x, mean=0., sigma=1.):
    _dnorm = grad(_pnorm)
    grads = auto_map(_dnorm, x, mean, sigma)
    return grads


def pnorm(x, mean=0., sigma=1.):
    p = auto_map(_pnorm, x, mean, sigma)
    return p


@jit
def _pnorm(x, mean=0., sigma=1.):
    scaled = (x - mean) / sigma
    return ndtr(scaled)


def qnorm(q, mean=0., sigma=1.):
    x = auto_map(_qnorm, q, mean, sigma)
    return x


@jit
def _qnorm(q, mean=0., sigma=1.):
    x = ndtri(q)
    return x * sigma + mean


def rnorm(key, mean=0., sigma=1., sample_shape=()):
    return _rnorm(key, sample_shape) * sigma + mean


def _rnorm(key, sample_shape=()):
    return jrand.normal(key, sample_shape)
