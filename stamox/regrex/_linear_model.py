import functools as ft

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import lax, vmap, grad, jit
import equinox as eqx

class RegressionModel(eqx.Module):

    def __init__(self) -> None:
        super().__init__()
    

    def set_method(self, method="ols"):
        pass

    def set_algorithm(self, alg="linalg"):
        pass    

    def __call__(self):
        pass


class OLS(RegressionModel):

    def __init__(self) -> None:
        super().__init__()
    

    def __call__(self, X, y):
        pass


@jtu.Partial(jit, static_argnames=('method'))
def _lm_linalg(X, y, method='pinv'):
    if method == 'pinv':
        X_pinv = jnp.linalg.pinv(X)
        params = jnp.dot(X_pinv, y)
    elif method == 'qr':
        Q, R = jnp.linalg.qr(X)
        params = jnp.linalg.solve(R, jnp.dot(Q.T, y))
    else:
        params, _, _, _ = jnp.linalg.lstsq(X, y, rcond=1)
    return params
