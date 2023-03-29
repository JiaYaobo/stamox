import functools as ft

import jax.numpy as jnp
from jax import grad, jit, vmap


def squared_difference(x, y):
    return jnp.square(x - y)


def dtriangular(x, low, high, peak):
    x = jnp.asarray(x)
    x = jnp.atleast_1d(x)
    _dΔ = grad(_ptriangular)
    grads = vmap(_dΔ, in_axes=(0, None, None, None))(x, low, high, peak)
    return grads


def ptriangular(x, low, high, peak):
    x = jnp.asarray(x)
    x = jnp.atleast_1d(x)
    p = vmap(_ptriangular, in_axes=(0, None, None, None))(x, low, high, peak)
    return p


@ft.partial(
    jit,
    static_argnames=(
        "low",
        "high",
        "peak",
    ),
)
def _ptriangular(x, low, high, peak):
    interval_length = high - low

    cond0 = (x >= low) & (x <= peak)
    result_inside_interval = jnp.where(
        cond0,
        squared_difference(x, low) / (interval_length * (peak - low)),
        1.0 - squared_difference(high, x) / (interval_length * (high - peak)),
    )

    cond1 = x < low
    result_if_not_big = jnp.where(cond1, jnp.zeros_like(x), result_inside_interval)

    cond2 = x >= high
    return jnp.where(cond2, jnp.ones_like(x), result_if_not_big)
