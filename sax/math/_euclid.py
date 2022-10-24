import functools as ft

import jax
import jax.numpy as jnp
from jax import jit

@ft.partial(jit, static_argnums=[1])
def _row_norms(x, squared=False):
    norms = jnp.einsum('ij, ij->i', x, x)

    if not squared:
        norms = jnp.sqrt(norms)
    return norms


@ft.partial(jit, static_argnums=[2, 3, 4])
def _euclidean_distances(x, y, x_norm_squared=None, y_norm_squared=None, squared=False):
    """Computational part of euclidean_distances

    Assumes inputs are already checked.

    If norms are passed as float32, they are unused. If arrays are passed as
    float32, norms needs to be recomputed on upcast chunks.
    """
    if x_norm_squared is not None:
        xx = x_norm_squared.reshape(-1, 1)
    else:
        xx = _row_norms(x, squared=True)[:, jnp.newaxis]

    if y is x:
        yy = xx.T
    else:
        if y_norm_squared is not None:
            yy = y_norm_squared.reshape(1, -1)
        else:
            yy = _row_norms(y, squared=True)[jnp.newaxis, :]

    distances = -2 * jnp.dot(x, y.T)
    distances += xx
    distances += yy
    distances = jnp.maximum(distances, 0)

    if squared:
        return distances
    else:
        return jnp.sqrt(distances)
