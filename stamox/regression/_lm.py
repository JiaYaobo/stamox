from typing import List, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike
from pandas import DataFrame

from ..core import make_partial_pipe
from ..distribution import pF, pt
from ..formula import get_design_matrices
from ._base import RegState


class OLSState(RegState):
    _X_names: List[str]
    _y_names: List[str]
    _SSE: float
    _SSR: float
    _SST: float
    _coef_std: ArrayLike

    def __init__(
        self,
        in_features,
        out_features,
        coefs,
        df_resid,
        df_model,
        n_obs,
        resid,
        fitted_values,
        rank,
        SSE,
        SSR,
        SST,
        coefs_std,
        X_names=None,
        y_names=None,
        dtype=jnp.float32,
        name="OLSResult",
    ):
        super().__init__(
            in_features,
            out_features,
            coefs,
            df_resid,
            df_model,
            n_obs,
            resid,
            fitted_values,
            rank,
            dtype=dtype,
            name=name,
        )
        self._X_names = X_names
        self._y_names = y_names
        self._SSE = SSE
        self._SSR = SSR
        self._SST = SST
        self._coef_std = coefs_std

    @property
    def X_names(self):
        return self._X_names

    @property
    def y_names(self):
        return self._y_names

    @property
    def SSE(self):
        return self._SSE

    @property
    def SSR(self):
        return self._SSR

    @property
    def SST(self):
        return self._SST

    @property
    def F_value(self):
        return (self.SSR / self._df_model) / (self.SSE / self._df_resid)

    @property
    def F_pvalue(self):
        return 1 - pF(self.F_value, self._df_model, self._df_resid, dtype=self.dtype)

    @property
    def R2(self):
        return self.SSR / self.SST

    @property
    def R2_adj(self):
        return 1 - (self.SSE / self._df_resid) / (self.SST / (self._n_obs - 1))

    @property
    def coef_std(self):
        return self._coef_std

    @property
    def StdErr(self):
        return self.coef_std * self.resid_stderr

    @property
    def t_values(self):
        return jnp.divide(self.coefs, self.StdErr)

    @property
    def t_pvalues(self):
        return (
            pt(
                jnp.abs(self.t_values),
                self.df_resid,
                lower_tail=False,
                dtype=self.dtype,
            )
            * 2
        )

    @property
    def resid_stderr(self):
        return jnp.sqrt(self.SSE / self.df_resid)


@make_partial_pipe
def lm(
    data: Union[List, Tuple, DataFrame, ArrayLike],
    formula=None,
    subset=None,
    weights=None,
    NA_action="drop",
    method="qr",
    dtype=jnp.float32,
) -> OLSState:
    """Fits a linear model using the given data and parameters.

    Args:
        data (Union[List, Tuple, DataFrame, ArrayLike]): The data to fit the linear model with.
        formula (str, optional): A formula for the linear model. Defaults to None.
        subset (list, optional): A list of indices to use as a subset of the data. Defaults to None.
        weights (array-like, optional): An array of weights to apply to the data. Defaults to None.
        NA_action (str, optional): The action to take when encountering missing values. Defaults to "drop".
        method (str, optional): The method to use for fitting the linear model. Defaults to "qr".
        dtype (jnp.float32, optional): The data type to use for the linear model. Defaults to jnp.float32.

    Returns:
        OLSState: The state of the fitted linear model.

    Example:
        >>> from stamox.regression import lm
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = np.random.uniform(size=(1000, 3))
        >>> y = (
                3 * X[:, 0]
                + 2 * X[:, 1]
                - 7 * X[:, 2]
                + np.random.standard_t(10, size=(1000,))
            )
        >>> data = pd.DataFrame(
                np.concatenate([X, y.reshape((-1, 1))], axis=1),
                columns=["x1", "x2", "x3", "y"],
            )
        >>> res = lm(data, "y ~ x1 + x2 + x3")
        >>> res.coefs
        Array([ 1., 3.,  2., -7.], dtype=float32)

    """
    if formula is None:
        if isinstance(data, DataFrame):
            raise ValueError("formula is required when data is DataFrame")
        elif isinstance(data, list):
            y = data[0]
            X = data[1]
        elif isinstance(data, tuple):
            y = data[0]
            X = data[1]
        elif isinstance(data, ArrayLike):
            y = data[:, 0]
            X = data[:, 1:]
        else:
            raise ValueError("data is not supported")
        X_names = None
        y_names = None
    else:
        matrices = get_design_matrices(data, formula, NA_action=NA_action, dtype=dtype)
        y = matrices.y
        X = matrices.X
        X_names = matrices.X_names
        y_names = matrices.y_names
    if subset is not None:
        y = y[subset, :]
        X = X[subset, :]
    if weights is not None:
        weights = jnp.asarray(weights).reshape(-1, 1)
        W = jnp.diag(weights.reshape(-1))
        y = jnp.multiply(y, weights)
        X = jnp.matmul(X.T, W).T
    return _fit_lm(X, y, method, X_names=X_names, y_names=y_names)


@eqx.filter_jit
def _fit_lm(X, y, method, X_names=None, y_names=None):
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
    if method == "inv":
        X_pinv = jnp.linalg.inv(X.T @ X) @ X.T
    elif method == "qr":
        Q, R = jnp.linalg.qr(X)
        R_inv = jnp.linalg.inv(R)
        X_pinv = R_inv @ Q.T
    elif method == "svd":
        U, S, Vt = jnp.linalg.svd(X)
        D = jnp.zeros((num_samples, in_features))
        D = D.at[:in_features, :in_features].set(jnp.diag(1 / S))
        X_pinv = Vt.T @ D.T @ U.T
    else:
        raise NotImplementedError("Not Implemented")
    coefs = X_pinv @ y
    _coefs_std = jnp.sqrt(jnp.diag(X_pinv @ X_pinv.T)).reshape(-1, 1)
    _fitted_values = X @ coefs
    _residuals = y - _fitted_values
    _df_resid = num_samples - in_features
    _df_model = in_features - 1
    _rank = jnp.linalg.matrix_rank(X)
    _SSE = jnp.sum(_residuals**2)
    _SST = jnp.sum((y - jnp.mean(y)) ** 2)
    _SSR = _SST - _SSE
    state = OLSState(
        in_features=in_features,
        out_features=out_features,
        coefs=coefs,
        df_resid=_df_resid,
        df_model=_df_model,
        n_obs=num_samples,
        resid=_residuals,
        fitted_values=_fitted_values,
        rank=_rank,
        SSE=_SSE,
        SSR=_SSR,
        SST=_SST,
        coefs_std=_coefs_std,
        dtype=X.dtype,
        X_names=X_names,
        y_names=y_names,
    )
    return state
