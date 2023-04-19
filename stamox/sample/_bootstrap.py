from typing import Callable, TypeVar

import jax.random as jrandom
from equinox import filter_jit, filter_vmap
from jaxtyping import ArrayLike, PyTree


ReturnValue = TypeVar("ReturnValue")


def bootstrap_sample(
    data: ArrayLike, num_samples: int, *, key: jrandom.KeyArray = None
) -> ArrayLike:
    """Generates `num_samples` bootstrap samples from `data` with replacement.

    Args:
        data (array-like): The original data.
        num_samples (int): The number of bootstrap samples to generate.
        key (jrandom.KeyArray, optional): A random key array. Defaults to None.

    Returns:
        ArrayLike: An array of size `(num_samples, len(data))` containing the bootstrap samples.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jrandom
        >>> from stamox.sample import bootstrap_sample
        >>> data = jnp.arange(10)
        >>> key = jrandom.PRNGKey(0)
        >>> bootstrap_sample(data, num_samples=3, key=key)
        Array([[9, 1, 6, 2, 9, 3, 9, 9, 4, 5],
                [4, 0, 4, 4, 6, 2, 5, 6, 5, 3],
                [7, 6, 9, 0, 0, 7, 0, 5, 8, 4]], dtype=int32)
    """
    # Determine the number of elements in the data
    n = data.shape[0]
    keys = jrandom.split(key, num_samples)

    @filter_jit
    def sample_fn(key: jrandom.KeyArray):
        # Draw n random indices from the data, with replacement
        sample_indices = jrandom.choice(key, n, (n,), replace=True)
        # Use the drawn indices to create a new bootstrap sample
        sample = data[sample_indices, ...]
        return sample

    samples = filter_vmap(sample_fn)(keys)
    return samples


def bootstrap(
    data: ArrayLike,
    call: Callable[..., ReturnValue],
    num_samples: int,
    *,
    key: jrandom.KeyArray = None
) -> PyTree:
    """Generates `num_samples` bootstrap samples from `data` with replacement, and calls `call` on each sample.

    Args:
        data (array-like): The original data.
        call (Callable[..., ReturnValue]): The function to call on each bootstrap sample.
        num_samples (int): The number of bootstrap samples to generate.
        key (jrandom.KeyArray, optional): A random key array. Defaults to None.

    Returns:
        PyTree: The return value of `call` on each bootstrap sample.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jrandom
        >>> from stamox.sample import bootstrap
        >>> data = jnp.arange(10)
        >>> bootstrap(data, jnp.mean, 3, key=key)
        Array([5.7000003, 3.9      , 4.6      ], dtype=float32)

    """
    samples = bootstrap_sample(data, num_samples=num_samples, key=key)
    return filter_jit(filter_vmap(lambda x: call(x)))(samples)
