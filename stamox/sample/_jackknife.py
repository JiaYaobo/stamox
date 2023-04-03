from typing import Callable, TypeVar

import jax.numpy as jnp
from equinox import filter_jit, filter_vmap
from jaxtyping import ArrayLike, PyTree

from ..core import make_partial_pipe


ReturnValue = TypeVar("ReturnValue")


@make_partial_pipe(name="JackknifeSampler")
@filter_jit
def jackknife_sample(data: ArrayLike):
    """Generates `num_samples` jackknife samples from `data` with replacement.

    Args:
        data (array-like): The original data.
        num_samples (int): The number of jackknife samples to generate.
        key (jrandom.KeyArray, optional): A random key array. Defaults to None.

    Returns:
        numpy.ndarray: An array of size (num_samples, len(data)) containing the jackknife samples.
    """
    n = jnp.shape(data)[0]

    @filter_jit
    def apply_except_one(x, i):
        idx = jnp.arange(x.shape[0])
        idx = jnp.where(idx >= i, idx + 1, idx)[:-1]
        return x[idx]

    samples = filter_vmap(apply_except_one, in_axes=(None, 0))(
        data, jnp.arange(n)
    )

    return samples


@make_partial_pipe(name="Jackknife")
def jackknife(
    data: ArrayLike, call: Callable[..., ReturnValue]
) -> PyTree:
    samples = jackknife_sample(data)
    return filter_jit(filter_vmap(call)(samples))
