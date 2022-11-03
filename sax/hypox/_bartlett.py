import functools as ft

import jax.numpy as jnp
from jax import jit, vmap

def bartlett(samples):
    return _bartlett(samples)

@jit
def _bartlett(samples):
    samples = jnp.array(samples)
    k = samples.shape[0]
    Ni = vmap(jnp.size, in_axes=(0,))(samples)
    ssq = vmap(ft.partial(jnp.var, ddof=1), in_axes=(0,))(samples)
    Ntot = jnp.sum(Ni, axis=0)
    spsq = jnp.sum((Ni- 1) * ssq, axis=0) / (1.0 * (Ntot- k))
    numer = (Ntot * 1.0 - k) * jnp.log(spsq) - jnp.sum((Ni - 1.0)*jnp.log(ssq), axis=0)
    denom = 1.0 + 1.0 / (3 * (k - 1))  * ((jnp.sum(1.0/(Ni - 1.0), axis=0)) -
                                     1.0/(Ntot - k))
    T = numer / denom
    return T