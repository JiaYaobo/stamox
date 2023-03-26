import jax.numpy as jnp

from .core import make_partial_pipe


mean = make_partial_pipe(jnp.mean, name='mean')
std = make_partial_pipe(jnp.std, name='std')
var = make_partial_pipe(jnp.var, name='var')
median = make_partial_pipe(jnp.median, name='median')

@make_partial_pipe(name='scale')
def scale(x, axis=0):
    # calculate standardized x along axis
    return (x - mean(x, axis=axis)) / std(x, axis=axis)


