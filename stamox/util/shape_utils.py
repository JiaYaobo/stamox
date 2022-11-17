import jax.numpy as jnp

def zero_dim_to_1_dim_array(x, dtype=None):
    return jnp.atleast_1d(x)


