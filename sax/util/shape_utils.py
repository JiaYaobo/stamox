import jax.numpy as jnp

def scalar_to_1dim_array(x):
    if x.ndim == 0:
        return jnp.expand_dims(x, 0)
    else:
        return x