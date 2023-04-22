"""Test for RegState"""
import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import stamox.pipe_functions as PF
from stamox.core import Pipeable, predict
from stamox.regression import lm, RegState


def test_regstate_init():
    regstate = RegState(in_features=3, out_features=1)
    np.testing.assert_equal(regstate.in_features, 3)
    np.testing.assert_equal(regstate.out_features, 1)
    np.testing.assert_equal(regstate.coefs.shape, (3, 1))
    np.testing.assert_equal(regstate.intercept.shape, (1, 1))
    np.testing.assert_equal(regstate.params.shape, (3, 1))


def test_olsstate():
    X = np.random.uniform(size=(1000, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
    data = pd.DataFrame(
        np.concatenate([X, y.reshape((-1, 1))], axis=1),
        columns=["x1", "x2", "x3", "y"],
    )

    res = lm(data, "y ~ x1 + x2 + x3", dtype=jnp.float64)
    np.testing.assert_allclose(predict(X, res), y.reshape((-1, 1)), atol=1e-5)
    np.testing.assert_allclose(
        np.sum(np.multiply(res.resid, X), axis=1), np.zeros(1000), atol=1e-4
    )
    np.testing.assert_equal(res.df_model, 3)
    np.testing.assert_equal(res.df_resid, 996)


def test_olsstate_err_t():
    np.random.seed(0)
    X = np.random.uniform(size=(1000, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + np.random.standard_t(10, size=(1000,))
    data = pd.DataFrame(
        np.concatenate([X, y.reshape((-1, 1))], axis=1),
        columns=["x1", "x2", "x3", "y"],
    )
    res = lm(data, "y ~ x1 + x2 + x3", dtype=jnp.float64)
    res_smf = smf.ols("y ~ x1 + x2 + x3", data=data).fit()
    np.testing.assert_allclose(
        res.coefs.reshape(-1), res_smf.params.values.reshape(-1), atol=1e-5
    )
    np.testing.assert_allclose(res.F_value, res_smf.fvalue, atol=1e-5)
    np.testing.assert_allclose(
        res.StdErr.reshape(-1), res_smf.bse.values.reshape(-1), atol=1e-5
    )
    np.testing.assert_allclose(
        res.t_values.reshape(-1), res_smf.tvalues.values.reshape(-1), atol=1e-5
    )
    np.testing.assert_allclose(res.R2, res_smf.rsquared, atol=1e-5)
    np.testing.assert_allclose(res.R2_adj, res_smf.rsquared_adj, atol=1e-5)
    np.testing.assert_allclose(res.t_pvalues.reshape(-1), res_smf.pvalues, atol=1e-5)


def test_pipe_olsstate():
    X = np.random.uniform(size=(1000, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
    data = pd.DataFrame(
        np.concatenate([X, y.reshape((-1, 1))], axis=1),
        columns=["x1", "x2", "x3", "y"],
    )

    model = (Pipeable(data) >> PF.lm(formula="y ~ x1 + x2 + x3", dtype=jnp.float64))()
    trans_X = (Pipeable(X) >> predict(state=model))()
    np.testing.assert_allclose(trans_X, y.reshape((-1, 1)), atol=1e-5)
