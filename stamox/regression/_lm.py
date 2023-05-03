from typing import List, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jaxtyping import ArrayLike
from pandas import DataFrame

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
        if X_names is None:
            X_names = ["Intercept"] + [f"X{i}" for i in range(in_features)]
        self._X_names = X_names
        if y_names is None:
            y_names = [f"y{i}" for i in range(out_features)]
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
    def f_value(self):
        return (self.SSR / self._df_model) / (self.SSE / self._df_resid)

    @property
    def f_pvalue(self):
        return 1 - pF(self.f_value, self._df_model, self._df_resid, dtype=self.dtype)

    @property
    def r2(self):
        return self.SSR / self.SST

    @property
    def r2_adj(self):
        return 1 - (self.SSE / self._df_resid) / (self.SST / (self._n_obs - 1))

    @property
    def coef_std(self):
        return self._coef_std

    @property
    def std_err(self):
        return jnp.dot(self.coef_std, self.resid_stderr).ravel()

    @property
    def t_values(self):
        return jnp.divide(self.coefs.ravel(), self.std_err.ravel())

    @property
    def p_values(self):
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

    @property
    def confint(self):
        rec = jnp.array([-1, 1]) * pt(0.975, self.df_resid, dtype=self.dtype)
        return vmap(lambda c, ci, z: c + ci * z, in_axes=(0, 0, None))(
            self.coefs,
            self.std_err,
            rec,
        )

    def _summary(self):
        summary_header = """
        Linear Regression Model Summary
        --------------------------------------------

        R-squared:          {r_sq:.4f}
        Adjusted R-squared: {adj_r_sq:.4f}
        F-statistic:        {f_value:.4f} on {df_model} and {df_resid} DF, p-value: {f_pvalue:.4f}
        SSE :               {SSE:.4f}
        SSR :               {SSR:.4f}
        Coefficients:
        {:<14} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}
        """.format(
            "",
            "Estimate",
            "Std.Error",
            "t-value",
            "p-value",
            "0.025",
            "0.975",
            r_sq=self.r2,
            adj_r_sq=self.r2_adj,
            f_value=self.f_value,
            df_model=self.df_model,
            df_resid=self.df_resid,
            f_pvalue=self.f_pvalue,
            SSE=self.SSE,
            SSR=self.SSR,
        )  # noqa: E501
        print(summary_header)
        new_line = "        " + "-" * 80
        print(new_line)
        for i, name in enumerate(self.X_names):
            p_value_formatted = (
                f"{self.p_values[i]:.4f}"
                if self.p_values[i] >= 0.001
                else f"{self.p_values[i]:.1e}"
            )
            summary_row = "        {:<14} {:>10.4f} {:>10.4f} {:>10.4f} {:>10} {:>10.4f} {:>10.4f}".format(
                name,
                self.coefs[i],
                self.std_err[i],
                self.t_values[i],
                p_value_formatted,
                self.confint[i, 0],
                self.confint[i, 1],
            )
            print(summary_row)


def lm(
    data: Union[List, Tuple, DataFrame, ArrayLike],
    formula: str = None,
    subset=None,
    weights=None,
    NA_action="drop",
    method="qr",
    dtype: jnp.dtype = jnp.float_,
) -> OLSState:
    """Fits a linear model using the given data and parameters.

    Args:
        data (Union[List, Tuple, DataFrame, ArrayLike]): The data to fit the linear model, if not Dataframe, must be [y, X].
        formula (str, optional): A formula for the linear model. Defaults to None.
        subset (list, optional): A list of indices to use as a subset of the data. Defaults to None.
        weights (array-like, optional): An array of weights to apply to the data. Defaults to None.
        NA_action (str, optional): The action to take when encountering missing values. Defaults to "drop".
        method (str, optional): The method to use for fitting the linear model. Defaults to "qr".
        dtype (jnp.float_, optional): The data type to use for the linear model. Defaults to jnp.float_.

    Returns:
        OLSState: The state of the fitted linear model.

    Example:
        >>> from stamox.functions import lm
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = np.random.uniform(size=(1000, 3))
        >>> y = (
                3 * X[:, 0]
                + 2 * X[:, 1]
                - 7 * X[:, 2]
                + 1.
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
        elif isinstance(data, jnp.ndarray) or isinstance(data, np.ndarray):
            y = data[:, 0]
            X = data[:, 1:]
        else:
            raise ValueError("data is not supported")

        if dtype is None:
            dtype = jnp.promote_types(y.dtype, X.dtype)
        y = jnp.asarray(y, dtype=dtype)
        X = jnp.asarray(X, dtype=dtype)
        X_names = None
        y_names = None
    else:
        if not isinstance(data, DataFrame):
            raise ValueError("data must be a DataFrame when formula is not None")
        matrices = get_design_matrices(data, formula, NA_action=NA_action, dtype=dtype)
        y = matrices.y
        X = matrices.X
        X_names = matrices.X_names
        y_names = matrices.y_names
    if subset is not None:
        y = y[subset, :]
        X = X[subset, :]
    if weights is not None:
        weights = jnp.asarray(weights, dtype=dtype).reshape(-1, 1)
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
        D = jnp.zeros((num_samples, in_features), dtype=X.dtype)
        D = D.at[:in_features, :in_features].set(jnp.diag(1 / S))
        X_pinv = Vt.T @ D.T @ U.T
    else:
        raise NotImplementedError(f"Not Implemented method {method}")
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
