from typing import Optional

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..core import make_partial_pipe
from ._pearson import pearsonr
from ._spearman import spearmanr


@make_partial_pipe
def cor(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    axis: int = 0,
    method: str = "pearson",
    dtype=jnp.float32,
) -> ArrayLike:
    """Calculates correlation between two arrays.

    Args:
        x (ArrayLike): The first array.
        y (Optional[ArrayLike], optional): The second array. Defaults to None.
        axis (int, optional): Axis along which the correlation is calculated. Defaults to 0.
        method (str, optional): Method used for calculating correlation. Defaults to "pearson".
        dtype (jnp.float32, optional): Data type of the output array. Defaults to jnp.float32.

    Returns:
        ArrayLike: Correlation between two arrays.

    Raises:
        NotImplementedError: If the specified method is not supported.

    Examples:
        >>> import jax.numpy as jnp
        >>> from stamox.correlation import cor
        >>> x = jnp.array([1, 2, 3, 4, 5])
        >>> y = jnp.array([5, 6, 7, 8, 7])
        >>> cor(x, y)
        Array(0.8320503, dtype=float32)
    """
    if method == "pearson":
        return pearsonr(x, y, axis, dtype)
    elif method == "spearman":
        return spearmanr(x, y, axis, dtype)
    else:
        raise NotImplementedError(f"method {method} not supported")
