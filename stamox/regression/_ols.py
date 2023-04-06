from typing import List, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax.random import KeyArray
from jaxtyping import ArrayLike, Bool

from ..core import Functional
from ._base import RegState


class OLSState(RegState):
    def __init__(self, in_features, out_features, use_intercept, *, key):
        super().__init__(in_features, out_features, use_intercept, key=key)


class OLS(Functional):
    use_intercept: Bool
    method: str
    key: KeyArray

    def __init__(self, use_intercept=True, method="inv", *, key=None):
        super().__init__(name="OLS", fn=None)
        self.use_intercept = use_intercept
        self.method = method
        self.key = key

    def fit(self, X, y) -> OLSState:
        return self._fit_ols(X, y)

    def __call__(self, Xy: Union[Tuple, List, ArrayLike]) -> OLSState:
        if isinstance(Xy, tuple) or isinstance(Xy, list):
            X = Xy[0]
            y = Xy[1]
        elif isinstance(Xy, ArrayLike):
            X = Xy[:, :-1]
            y = Xy[:, -1]
        return self._fit_ols(X, y)

    @eqx.filter_jit
    def _fit_ols(self, X, y):
        num_samples = X.shape[0]
        in_features = X.shape[1]
        out_features = 1
        if len(jnp.shape(y)) == 2:
            if jnp.shape(y)[1] > 1:
                raise NotImplementedError("multiple output is not supported yet.")
        elif len(jnp.shape(y)) == 1:
            y = jnp.atleast_1d(y)
        else:
            raise NotImplementedError("batchable output is not supported yet.")
        if self.use_intercept:
            ones = jnp.ones((num_samples, 1))
            X = jnp.hstack([ones, X])
        state = OLSState(in_features, out_features, self.use_intercept, key=self.key)
        if self.method == "inv":
            X_pinv = jnp.linalg.inv(X.T @ X) @ X.T
        elif self.method == "qr":
            Q, R = jnp.linalg.qr(X)
            R_inv = jnp.linalg.inv(R)
            X_pinv = R_inv @ Q.T
        elif self.method == "svd":
            U, S, Vt = jnp.linalg.svd(X)
            D = jnp.zeros((num_samples, in_features))
            D[:in_features, :in_features] = jnp.diag(1 / S)
            X_pinv = Vt.T @ D.T @ U.T
        else:
            raise NotImplementedError("Not Implemented")
        coefs = X_pinv @ y
        if self.use_intercept:
            weight, bias = coefs[1:], coefs[0]
        else:
            weight = coefs
        state = eqx.tree_at(lambda x: x.linear.weight, state, weight)
        if self.use_intercept:
            state = eqx.tree_at(lambda x: x.linear.bias, state, bias)
        return state
