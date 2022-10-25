import functools as ft

import jax
import jax.numpy as jnp
from jax import jit


@ft.partial(jit, static_argnums=[2])
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
