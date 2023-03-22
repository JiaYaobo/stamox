import jax.numpy as jnp
from jax import jit, lax

from ..maps import auto_map


def drademacher(k):
    dens = auto_map(_drademacher, k)
    return dens


@jit
def _drademacher(k):
    in_support = jnp.logical_or(k == -1., k == 1.)
    dens = jnp.where(in_support, 1./2., 0)
    return dens


def prademacher(k):
    p = auto_map(_prademacher, k)
    return p


@jit
def _prademacher(k):
    cond0 = k < -1.
    cond1 = jnp.logical_and(k >= -1., k < 1.)
    cond2 = k >= 1.
    index = jnp.argwhere(jnp.array([cond0, cond1, cond2]), size=1).squeeze()
    branches = [lambda: 0., lambda: 1/2, lambda: 1.]
    p = lax.switch(index, branches, k)
    return p
