from functools import partial

import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnames=('log_d'))
def log_d(dens, log_d=False):

    if log_d:
        return jnp.log(dens)
    return dens
