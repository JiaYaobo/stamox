from typing import Callable, TypeVar

import jax.numpy as jnp
from equinox import filter_jit, filter_vmap
from jaxtyping import ArrayLike, PyTree


ReturnValue = TypeVar("ReturnValue")


def jackknife_sample(data: ArrayLike) -> ArrayLike:
    """Generates `num_samples` jackknife samples from `data` with replacement.

    Args:
        data (array-like): The original data.

    Returns:
        ArrayLike: An array of size (len(data)-1, len(data)) containing the jackknife samples.

    Example:
        >>> import jax.numpy as jnp
        >>> from stamox.sample import jackknife_sample
        >>> data = jnp.arange(3)
        >>> jackknife_sample(data)
        Array([[1, 2],
                [0, 2],
                [0, 1]], dtype=int32)

    """
    n = jnp.shape(data)[0]

    @filter_jit
    def apply_except_one(x, i):
        idx = jnp.arange(x.shape[0])
        idx = jnp.where(idx >= i, idx + 1, idx)[:-1]
        return x[idx]

    samples = filter_vmap(apply_except_one, in_axes=(None, 0))(data, jnp.arange(n))

    return samples


def jackknife(data: ArrayLike, call: Callable[..., ReturnValue]) -> PyTree:
    """Computes the jackknife estimate of a given data set.

    Args:
        data (ArrayLike): The data set to be analyzed.
        call (Callable[..., ReturnValue]): A function to be applied to each sample.

    Returns:
        PyTree: The jackknife estimate of the data set.

    Example:
        >>> import jax.numpy as jnp
        >>> from stamox.sample import jackknife
        >>> data = jnp.arange(3)
        >>> jackknife(data, lambda x: jnp.mean(x))
        Array([1.5, 1. , 0.5], dtype=float32)
    """
    samples = jackknife_sample(data)
    return filter_jit(filter_vmap(lambda x: call(x)))(samples)
