import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit, grad

from ..util import zero_dim_to_1_dim_array


def dlaplace(x, loc=0., scale=1.):
    x = jnp.asarray(x)
    x = zero_dim_to_1_dim_array(x)
    _dlaplace = grad(_plaplace)
    grads = vmap(_dlaplace, in_axes=(0, None, None))(x, loc, scale)
    return grads


def plaplace(x,  loc=0., scale=1.):
    x = jnp.asarray(x)
    x = zero_dim_to_1_dim_array(x)
    p = vmap(_plaplace, in_axes=(0, None, None))(x, loc, scale)
    return p


@ft.partial(jit, static_argnames=("loc", "scale",))
def _plaplace(x, loc=0., scale=1.):
    scaled = (x - loc) / scale
    return 0.5 - 0.5 * jnp.sign(scaled) * jnp.expm1(-jnp.abs(scaled))


def qlaplace(q, loc=0., scale=1.):
    q = jnp.asarray(q)
    q = zero_dim_to_1_dim_array(q)
    q = vmap(_qlaplace, in_axes=(0, None, None))(q, loc, scale)
    return q


@ft.partial(jit, static_argnames=("loc", "scale",))
def _qlaplace(q, loc=0., scale=1.):
    a = q - 0.5
    return loc - scale * jnp.sign(a) * jnp.log1p(-2 * jnp.abs(a))


def rlaplace(key, loc=0., scale=1., sample_shape=()):
    return _rlaplace(key, sample_shape) * scale + loc


def _rlaplace(key, sample_shape=()):
    return jrand.laplace(key, sample_shape)
