from typing import Optional

import jax.numpy as jnp
from equinox import filter_jit
from jax.scipy.stats import rankdata
from jaxtyping import ArrayLike


def spearmanr(x: ArrayLike, y: Optional[ArrayLike] = None, axis: int = 0) -> ArrayLike:
    """Calculates a Spearman rank-order correlation coefficient and the p-value to test for non-correlation.

    Args:
        x (ArrayLike): An array of values.
        y (Optional[ArrayLike], optional): An array of values. Defaults to None.
        axis (int, optional): The axis along which to calculate. Defaults to 0.

    Raises:
        ValueError: If the supplied axis argument is greater than 1 or if the number of dimensions of the array is greater than 2.

    Returns:
        A array-like containing the Spearman rank-order correlation coefficient and the p-value to test for non-correlation.

    Examples:
        >>> import jax.numpy as jnp
        >>> from stamox.correlation import spearmanr
        >>> x = jnp.array([1, 2, 3, 4, 5])
        >>> y = jnp.array([5, 6, 7, 8, 7])
        >>> spearmanr(x, y)
        Array(0.8207823038101196, dtype=float32)
    """
    if axis is not None and axis > 1:
        raise ValueError(
            "spearmanr only handles 1-D or 2-D arrays, "
            "supplied axis argument {}, please use only "
            "values 0, 1 or None for axis".format(axis)
        )
    x = jnp.asarray(x)

    if axis is None:
        x = x.ravel()
        axis_out = 0
    else:
        axis_out = axis

    if x.ndim > 2:
        raise ValueError("spearmanr only handles 1-D or 2-D arrays")

    if y is None:
        if x.ndim < 2:
            raise ValueError("`spearmanr` needs at least 2 " "variables to compare")
    else:
        y = jnp.asarray(y, dtype=x.dtype)
        if axis is None:
            y = y.ravel()
        if axis_out == 0:
            x = jnp.column_stack((x, y))
        else:
            x = jnp.row_stack((x, y))
    return _spearmanr(x, axis_out)


@filter_jit
def _spearmanr(x, axis):
    x_ranked = jnp.apply_along_axis(rankdata, axis, x)
    coef = jnp.corrcoef(x_ranked, rowvar=axis)
    if coef.shape == (2, 2):
        coef = coef[1, 0]
    return coef
