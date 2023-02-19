import jax.numpy as jnp
from jax import jit


@jit
def soft_threshold(rho, lamda):
    """Soft threshold function"""
    return jnp.multiply(jnp.abs(rho) , jnp.max(jnp.abs(rho) - lamda), 0)