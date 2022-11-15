import jax.numpy as jnp
import jax.tree_util as jtu
from jax import vmap, jit, lax

from ..util import zero_dim_to_1_dim_array


def drademacher(k):
    k = jnp.asarray(k)
    k = zero_dim_to_1_dim_array(k)
    dens = vmap(_drademacher)(k)
    return dens


@jit
def _drademacher(k):
    in_support = jnp.logical_or(k == -1., k == 1.)
    dens = jnp.where(in_support, 1./2., 0)
    return dens


def prademacher(k):
    k = jnp.asarray(k)
    k = zero_dim_to_1_dim_array(k)
    p = vmap(_prademacher)(k)
    return p


def _prademacher(k):
    cond0 = k < -1.
    cond1 = jnp.logical_and(k >= -1., k < 1.)
    cond2 = k >= 1.
    index = jnp.argwhere(jnp.array([cond0, cond1, cond2]), size=1).squeeze()
    branches = [lambda: 0., lambda: 1/2, lambda: 1.]
    p = lax.switch(index, branches, k)
    return p
