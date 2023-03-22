import functools as ft

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import lax, vmap, grad, jit
import equinox as eqx

from jaxtyping.array_types import AbstractArray
from jaxtyping.pytree_type import PyTree

from ._base import dispatch




def wls(X, y, weights=None, method='pinv'):
    x = X * jnp.sqrt(weights) * x
    pass

def ols(X, y, method):
    func = dispatch(method=method)
    params = func(X, y)
    return params
    