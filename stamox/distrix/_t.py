import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import jit, vmap, grad
from jax.scipy.special import betainc

from tensorflow_probability.substrates.jax.math import special as tfp_special


def dt(x, df, loc=0., scale=1.):
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = jnp.expand_dims(x, axis=0)
    _dt = grad(_pt)
    grads = vmap(_dt, in_axes=(0, None, None, None))(x, df, loc, scale)
    return grads


def pt(x, df, loc=0., scale=1.):
    x = jnp.asarray(x)
    p = vmap(_pt, in_axes=(0, None, None, None))(x, df, loc, scale)
    return p


@ft.partial(jit, static_argnames=("loc", "scale",))
def _pt(x, df, loc=0., scale=1.):
    scaled = (x - loc) / scale
    scaled_squared = scaled * scaled
    beta_value = df / (df + scaled_squared)

    # when scaled < 0, returns 0.5 * Beta(df/2, 0.5).cdf(beta_value)
    # when scaled > 0, returns 1 - 0.5 * Beta(df/2, 0.5).cdf(beta_value)
    return 0.5 * (
        1
        + jnp.sign(scaled)
        - jnp.sign(scaled) * betainc(0.5 * df, 0.5, beta_value)
    )


def qt(q, df, loc=0., scale=1.):
    q = jnp.asarray(q)
    q = vmap(_qt, in_axes=(0, None, None, None))(q, df, loc, scale)
    return q


@ft.partial(jit, static_argnames=("loc", "scale",))
def _qt(q, df, loc=0., scale=1.):
    beta_value = tfp_special.betaincinv(0.5 * df, 0.5, 1 - jnp.abs(1 - 2 * q))
    scaled_squared = df * (1 / beta_value - 1)
    scaled = jnp.sign(q - 0.5) * jnp.sqrt(scaled_squared)
    return scaled * scale + loc


def rt(key, df, loc=0., scale=1., sample_shape=()):
    return _rt(key, df, sample_shape) * scale + loc


def _rt(key, df, sample_shape=()):
    return jrand.t(key, df, sample_shape)
