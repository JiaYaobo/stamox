import jax.numpy as jnp
from jax import vmap

from .core import make_partial_pipe


mean = make_partial_pipe(jnp.mean, name="mean")
std = make_partial_pipe(jnp.std, name="std")
var = make_partial_pipe(jnp.var, name="var")
median = make_partial_pipe(jnp.median, name="median")


@make_partial_pipe(name="scale")
def scale(x, axis=0):
    # calculate standardized x along axis
    _mean = mean(x, axis=axis)
    _std = std(x, axis=axis, ddof=1)
    _scaled = vmap(lambda a, b, c: (a - b) / c, in_axes=(axis, None, None))(
        x, _mean, _std
    )
    return _scaled
