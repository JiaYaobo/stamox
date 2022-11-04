import jax.numpy as jnp

def zero_dim_to_1_dim_array(x):
    if x.ndim == 0:
        return jnp.expand_dims(x, 0)
    else:
        return x