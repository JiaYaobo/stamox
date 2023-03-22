import jax.numpy as jnp
import jax.random as jrand
from jax import jit, grad


from ..maps import auto_map


def dcauchy(x, loc=0., scale=1.):
    _dcauchy = grad(_pcauchy)
    grads = auto_map(_dcauchy, x, loc, scale)
    return grads


def pcauchy(x,  loc=0., scale=1.):
    p = auto_map(_pcauchy, x, loc, scale)
    return p


@jit
def _pcauchy(x, loc=0., scale=1.):
    scaled = (x - loc) / scale
    return jnp.arctan(scaled) / jnp.pi + 0.5


def qcauchy(q, loc=0., scale=1.):
    x = auto_map(_qcauchy, q, loc, scale)
    return x


@jit
def _qcauchy(q, loc=0., scale=1.):
    return loc + scale * jnp.tan(jnp.pi * (q - 0.5))


def rcauchy(key, loc=0., scale=1., sample_shape=()):
    return _rcauchy(key, sample_shape) * scale + loc


def _rcauchy(key, sample_shape=()):
    return jrand.cauchy(key, sample_shape)
