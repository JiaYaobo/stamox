import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import equinox as eqx


from ..base import Model


class Corr(Model):

    def __init__(self) -> None:
        super().__init__()

    
    def __call__(self, x, y, tolerance=1e-8):
        nobs, k_yvar = y.shape
        nobs, k_xvar = x.shape

        k = jnp.min([k_yvar, k_xvar])

        x = jnp.array(x)
        y = jnp.array(y)

        x = x - x.mean()
        y = y - y.mean()

        ux, sx, vx = jsp.linalg.svd(x, 0)
        # vx_ds = vx.T divided by sx
        vx_ds = vx.T
        uy, sy, vy = jsp.linalg.svd(y, 0)
        # vy_ds = vy.T divided by sy
        vy_ds = vy.T
        u, s, v = jsp.linalg.svd(ux.T.dot(uy), 0)

        # Correct any roundoff
        corr = jnp.array([max(0, min(s[i], 1)) for i in range(len(s))])

        x_coef = vx_ds.dot(u[:, :k])
        y_coef = vy_ds.dot(v.T[:, :k])

        return (corr, x_coef, y_coef)

