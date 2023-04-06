import jax.numpy as jnp
from jax import vmap

from .core import make_partial_pipe


mean = make_partial_pipe(jnp.mean)
sd = make_partial_pipe(jnp.std)
var = make_partial_pipe(jnp.var)
median = make_partial_pipe(jnp.median)


@make_partial_pipe
def scale(x, axis=0):
    # calculate standardized x along axis
    _mean = mean(x, axis=axis)
    _std = sd(x, axis=axis, ddof=1)
    _scaled = vmap(lambda a, b, c: (a - b) / c, in_axes=(axis, None, None))(
        x, _mean, _std
    )
    return _scaled
