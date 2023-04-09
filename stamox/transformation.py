import jax.numpy as jnp
from jax import lax
from jaxtyping import ArrayLike

from .core import partial_pipe_jit


@partial_pipe_jit
def boxcox(x: ArrayLike, lmbda: ArrayLike) -> ArrayLike:
    """Computes the Box-Cox transformation of a given array.

    Args:
        x: An array-like object to be transformed.
        lmbda: An array-like object containing the lambda values for the transformation.

    Returns:
        The transformed array.

    Example:
        >>> from stamox.transformation import boxcox
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1, 2, 3, 4, 5], dtype=jnp.float32)
        >>> lmbda = jnp.array([0, 0, 0, 0, 0], dtype=jnp.float32)
        >>> boxcox(x, lmbda)
        Array([0.        , 0.6931472 , 1.0986123 , 1.3862944 , 1.6094378 ], dtype=float32)
    """
    return lax.select(lmbda == 0, jnp.log(x), (x ** lmbda - 1) / lmbda)




