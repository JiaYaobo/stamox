import functools as ft

import jax.numpy as jnp
import jax.random as jrand
from jax import jit, vmap, grad, lax

from stamox.util import zero_dim_to_1_dim_array

def squared_difference(x, y):
    return jnp.square(x - y)



def ptriangular(x, low, high, peak):
    x = jnp.asarray(x)
    x = zero_dim_to_1_dim_array(x)
    p = vmap(_ptriangular, in_axes=(0, None, None, None))(x, low, high, peak)
    return p


@ft.partial(jit, static_argnames=('low', 'high', 'peak', ))
def _ptriangular(x, low, high, peak):
    interval_length = high - low
    cond0 = (x >= low) & (x <= peak)
    result_inside_interval = lax.cond(cond0, 
                                squared_difference(x, low) / (interval_length * (peak - low)), 
                                1.0 - squared_difference(high, x) / (interval_length * (high - peak)))

    cond1 = x < low
    result_if_not_big = lax.cond(
        cond1, jnp.zeros_like(x), result_inside_interval)
    
    cond2 = x>= high
    return lax.cond(cond2, jnp.ones_like(x), result_if_not_big)