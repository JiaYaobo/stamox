import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import jit, vmap, grad


from stamox.util import zero_dim_to_1_dim_array


def dcauchy(x, loc=0., scale=1.):
    x = jnp.asarray(x)
    x = zero_dim_to_1_dim_array(x)
    _dcauchy = grad(_pcauchy)
    grads = vmap(_dcauchy, in_axes=(0, None, None))(x, loc, scale)
    return grads


def pcauchy(x,  loc=0., scale=1.):
    x = jnp.asarray(x)
    x = zero_dim_to_1_dim_array(x)
    p = vmap(_pcauchy, in_axes=(0, None, None, None))(x, loc, scale)
    return p


@ft.partial(jit, static_argnames=("loc", "scale",))
def _pcauchy(x, loc=0., scale=1.):
    scaled = (x - loc) / scale
    return jnp.arctan(scaled) / jnp.pi + 0.5


def qcauchy(q, loc=0., scale=1.):
    q = jnp.asarray(q)
    q = zero_dim_to_1_dim_array(q)
    q = vmap(_qcauchy, in_axes=(0, None, None))(q, loc, scale)
    return q


@ft.partial(jit, static_argnames=("loc", "scale",))
def _qcauchy(q, loc=0., scale=1.):
    return loc + scale * jnp.tan(jnp.pi * (q - 0.5))


def rcauchy(key, loc=0., scale=1., sample_shape=()):
    return _rcauchy(key, sample_shape) * scale + loc


def _rcauchy(key, sample_shape=()):
    return jrand.cauchy(key, sample_shape)
