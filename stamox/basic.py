import jax.numpy as jnp
from jax import vmap
from jaxtyping import ArrayLike


mean = jnp.mean
sd = jnp.std
var = jnp.var
median = jnp.median
quantile = jnp.quantile
min = jnp.min
max = jnp.max
sum = jnp.sum
prod = jnp.prod
cumsum = jnp.cumsum
cumprod = jnp.cumprod
diff = jnp.diff


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
