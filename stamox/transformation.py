import jax.numpy as jnp
from jax import lax
from jaxtyping import ArrayLike

from .core import partial_pipe_jit


@partial_pipe_jit
def boxcox(x: ArrayLike, lmbda: ArrayLike) -> ArrayLike:
    return lax.select(lmbda == 0, jnp.log(x), (x ** lmbda - 1) / lmbda)




