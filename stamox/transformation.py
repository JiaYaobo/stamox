import jax.numpy as jnp


def z_fisher(r=None, z=None):
    if z is None:
        return  jnp.arctanh(r)
    else:
        return jnp.tanh(z)