import jax.numpy as jnp
from jax import jit, grad

from ..maps import auto_map
from ._exp import rexp


def dpareto(x, scale, alpha):
    _dpareto = grad(_ppareto)
    grads = auto_map(_dpareto, x, scale, alpha)
    return grads


def rpareto(key, scale, alpha, sample_shape=()):
    y = rexp(key, alpha, sample_shape)
    return jnp.exp(y) * scale


def ppareto(x, scale, alpha):
    p = auto_map(_ppareto, x, scale, alpha)
    return p


def qpareto(q, scale, alpha):
    x = auto_map(_qpareto, q, scale, alpha)
    return x


@jit
def _ppareto(x, scale, alpha):
    return 1 - jnp.power(scale / x, alpha)


@jit
def _qpareto(q, scale, alpha):
    return scale / jnp.power(1 - q, 1 / alpha)
