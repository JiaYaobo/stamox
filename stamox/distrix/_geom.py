import functools as ft

import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
from jax import vmap, jit

from ..util import zero_dim_to_1_dim_array


def dgeom(k, p):
    k = jnp.asarray(k, dtype=jnp.int32)
    k = zero_dim_to_1_dim_array(k)
    pp = vmap(_dgeom, in_axes=(0, None))(k, p)
    return pp


@ft.partial(jit, static_argnames=('p', ))
def _dgeom(k, p):
    return jnp.power(1-p, k - 1) * p
