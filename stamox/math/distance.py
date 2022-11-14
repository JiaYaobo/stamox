import functools as ft

import jax
import jax.numpy as jnp
from jax import jit




@jit
def _mahalanobis_distance(x, Sinv, y=None, axis=0):
    mu_x = jnp.mean(x, axis=0)
    return jnp.sqrt((x - mu_x) @ Sinv @ (x - mu_x))


@ft.partial(jit, static_argnames=('p'))
def minkowski_distance_p(x, y, p=2):
    """Compute the pth power of the L**p distance between two arrays.
    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.
    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Examples
    --------
    >>> from scipy.spatial import minkowski_distance_p
    >>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
    array([2, 1])
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y, dtype=x.dtype)

    if p == 1:
        return jnp.sum(jnp.abs(y - x), axis=-1)
    elif p == jnp.inf:
        return jnp.amax(jnp.abs(y - x), axis=-1)
    else:
        return jnp.sum(jnp.abs(y - x)**p, axis=-1)


@ft.partial(jit, static_argnums=[2])
def minkowski_distance(x, y, p=2):
    """Compute the L**p distance between two arrays.
    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Examples
    --------
    >>> from sax.classix import minkowski_distance
    >>> minkowski_distance([[0,0],[0,0]], [[1,1],[0,1]])
    array([ 1.41421356,  1.        ])
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y, dtype=x.dtype)

    if p == jnp.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p)**(1./p)


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
