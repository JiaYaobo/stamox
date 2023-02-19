import jax.numpy as jnp
from jax import jit



@jit
def squared_difference(x, y):
    return jnp.square(x - y)