import functools as ft

import jax.numpy as jnp
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


@jit
def _lm_linalg(X, y):
    w_linalg = jnp.dot(jnp.dot(jnp.linalg.inv(jnp.dot*X.T, X), X.T), y)
    return w_linalg