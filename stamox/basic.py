import jax.numpy as jnp

from .core import make_partial_pipe


mean = make_partial_pipe(jnp.mean)
std = make_partial_pipe(jnp.std)
var = make_partial_pipe(jnp.var)
median = make_partial_pipe(jnp.median)


