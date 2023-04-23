import jax.numpy as jnp
from jax import vmap
from jaxtyping import ArrayLike

from .core import make_partial_pipe


mean = make_partial_pipe(jnp.mean, "mean")
sd = make_partial_pipe(jnp.std, "sd")
var = make_partial_pipe(jnp.var, "var")
median = make_partial_pipe(jnp.median, "median")
quantile = make_partial_pipe(jnp.quantile, "quantile")
min = make_partial_pipe(jnp.min, "min")
max = make_partial_pipe(jnp.max, "max")
sum = make_partial_pipe(jnp.sum, "sum")
prod = make_partial_pipe(jnp.prod, "prod")
cumsum = make_partial_pipe(jnp.cumsum, "cumsum")
cumprod = make_partial_pipe(jnp.cumprod, "cumprod")
diff = make_partial_pipe(jnp.diff, "diff")


@make_partial_pipe
def scale(x: ArrayLike, axis: int = 0, dtype=jnp.float_) -> ArrayLike:
    """Calculate standardized x along axis.

    Args:
        x (array-like): Input array.
        axis (int, optional): Axis along which to calculate mean and standard deviation. Defaults to 0.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: Standardized x along axis.
    """
    x = jnp.asarray(x, dtype=dtype)
    _mean = mean(x, axis=axis)
    _std = sd(x, axis=axis, ddof=1)
    _scaled = vmap(lambda a, b, c: (a - b) / c, in_axes=(axis, None, None))(
        x, _mean, _std
    )
    return _scaled
