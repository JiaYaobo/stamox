from typing import Optional

import jax.numpy as jnp
from equinox import filter_jit
from jaxtyping import ArrayLike

from ..core import make_partial_pipe


@make_partial_pipe
def pearsonr(
    x: ArrayLike, y: Optional[ArrayLike] = None, axis: int = 0, dtype=jnp.float32
) -> ArrayLike:
    """Computes Pearson correlation coefficient for two arrays.

    Args:
        x: An array-like object containing the first set of data.
        y: An optional array-like object containing the second set of data. If not
        provided, `x` is assumed to contain two sets of data.
        axis: The axis along which the correlation coefficient should be computed.
        Must be 0 or 1. Defaults to 0.
        dtype: The data type of the input arrays. Defaults to jnp.float32.

    Returns:
        An array-like object containing the Pearson correlation coefficient.

    Examples:
        >>> import jax.numpy as jnp
        >>> from stamox.correlation import pearsonr
        >>> x = jnp.array([1, 2, 3, 4, 5])
        >>> y = jnp.array([5, 6, 7, 8, 7])
        >>> pearsonr(x, y)
        Array(0.8320503, dtype=float32)
    """
    if axis is not None and axis > 1:
        raise ValueError(
            "pearsonr only handles 1-D or 2-D arrays, "
            "supplied axis argument {}, please use only "
            "values 0, 1 or None for axis".format(axis)
        )
    x = jnp.asarray(x, dtype=dtype)
    if axis is None:
        x = x.ravel()
        axis_out = 0
    else:
        axis_out = axis

    if x.ndim > 2:
        raise ValueError("pearsonr only handles 1-D or 2-D arrays")

    if y is None:
        if x.ndim < 2:
            raise ValueError("`pearsonr` needs at least 2 " "variables to compare")
    else:
        y = jnp.asarray(y, dtype=dtype)
        if axis is None:
            y = y.ravel()
        if axis_out == 0:
            x = jnp.column_stack((x, y))
        else:
            x = jnp.row_stack((x, y))

    return _pearsonr(x, axis)


@filter_jit
def _pearsonr(x, axis):
    coef = jnp.corrcoef(x, rowvar=axis)
    if coef.shape == (2, 2):
        coef = coef[1, 0]
    return coef
