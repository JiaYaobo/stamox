import jax.numpy as jnp
import jax.random as jrand
from jax import jit,  grad, vmap
from jax.scipy.special import ndtr, ndtri

from ..maps import auto_map, cube_map


@jit
def _pnorm(x, mean=0., sigma=1.):
    scaled = (x - mean) / sigma
    return ndtr(scaled)


_dnorm = jit(grad(_pnorm))


def qnorm(q, mean=0., sigma=1.):
    x = auto_map(_qnorm, q, mean, sigma)
    return x


@jit
def _qnorm(q, mean=0., sigma=1.):
    x = ndtri(q)
    return x * sigma + mean


def dnorm(x, mean=0., sigma=1.):
    # _dnorm = grad(_pnorm)
    grads = auto_map(_dnorm, x, mean, sigma)
    return grads


def pnorm(x, mean=0., sigma=1., lower_tail=True, log_prob=False, map_rule='auto'):

    if map_rule == 'auto':
        p = auto_map(_pnorm, x, mean, sigma)
    elif map_rule == 'cube':
        p = cube_map(_pnorm, x, mean, sigma)
    
    if lower_tail is False:
        p = vmap(lambda x: 1-x, in_axes=(0))(p)

    if log_prob is True:
        p = vmap(jnp.log, in_axes=(0))(p)

    return p


def rnorm(key, mean=0., sigma=1., sample_shape=()):
    return _rnorm(key, sample_shape) * sigma + mean


def _rnorm(key, sample_shape=()):
    return jrand.normal(key, sample_shape)
