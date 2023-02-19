import jax.numpy as jnp
import jax.random as jrand
from jax import jit, grad

from ..maps import auto_map


def dlaplace(x, loc=0., scale=1.):
    _dlaplace = grad(_plaplace)
    grads = auto_map(_dlaplace, x, loc, scale)
    return grads


def plaplace(x,  loc=0., scale=1.):
    p = auto_map(_plaplace, x, loc, scale)
    return p


@jit
def _plaplace(x, loc=0., scale=1.):
    scaled = (x - loc) / scale
    return 0.5 - 0.5 * jnp.sign(scaled) * jnp.expm1(-jnp.abs(scaled))


def qlaplace(q, loc=0., scale=1.):
    q = auto_map(_qlaplace, q, loc, scale)
    return q


@jit
def _qlaplace(q, loc=0., scale=1.):
    a = q - 0.5
    return loc - scale * jnp.sign(a) * jnp.log1p(-2 * jnp.abs(a))


def rlaplace(key, loc=0., scale=1., sample_shape=()):
    return _rlaplace(key, sample_shape) * scale + loc


def _rlaplace(key, sample_shape=()):
    return jrand.laplace(key, sample_shape)
