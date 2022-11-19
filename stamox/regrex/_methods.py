import functools as ft

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import lax, vmap, grad, jit


@jit
def _lm_pinv(X, y):
    X_pinv = jnp.linalg.pinv(X)
    params = jnp.dot(X_pinv, y)
    return params


@jit
def _lm_qr(X, y):
    Q, R = jnp.linalg.qr(X)
    params = jnp.linalg.solve(R, jnp.dot(Q.T, y))
    return params


