import functools as ft

import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import jit, vmap

import numpy as np

from ..base import Model


class Corr(Model):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x1, x2, tolerance=1e-8):
        _corr(x1, x2, tolerance=tolerance)


@ft.partial(jit, static_argnames=('tolerance', ))
def _corr(x1, x2, tolerance=1e-8):
    nobs, k_yvar = x1.shape
    nobs, k_xvar = x2.shape

    k = np.min([k_yvar, k_xvar])

    x = jnp.array(x1)
    y = jnp.array(x2)

    x = x - x.mean()
    y = y - y.mean()

    ux, sx, vx = jsp.linalg.svd(x)
    # vx_ds = vx.T divided by sx
    vx_ds = vx.T
    uy, sy, vy = jsp.linalg.svd(y)
    # vy_ds = vy.T divided by sy
    vy_ds = vy.T
    u, s, v = jsp.linalg.svd(ux.T.dot(uy))

    # Correct any roundoff
    corr = vmap(lambda c: jnp.maximum(0, jnp.minimum(c, 1)), in_axes=0)(s)

    x_coef = vx_ds.dot(u[:, :k])
    y_coef = vy_ds.dot(v.T[:, :k])

    return corr, x_coef, y_coef
