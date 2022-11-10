import jax
import jax.numpy as jnp
import jax.random as jr
import random
import equinox as eqx

from jaxtyping import Array



def _boot_splits(data: Array, times=25, strata=None, breaks=4, pool=0.1):
    n = data.shape[0]

    if strata == None:
        pass