"""Test for RegState"""
import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from absl.testing import absltest
from jax import config
from jax._src import test_util as jtest

from stamox.core import Pipeable
from stamox.regression import lm, RegState


config.update("jax_enable_x64", True)


class RegStateTest(jtest.JaxTestCase):
    def test_regstate_init(self):
        regstate = RegState(in_features=3, out_features=1)
        self.assertEqual(regstate.in_features, 3)
        self.assertEqual(regstate.out_features, 1)
        self.assertEqual(regstate.coefs.shape, (3, 1))
        self.assertEqual(regstate.intercept.shape, (1, 1))
        self.assertEqual(regstate.params.shape, (3, 1))

    def test_olsstate(self):
        X = np.random.uniform(size=(1000, 3))
        y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
        data = pd.DataFrame(
            np.concatenate([X, y.reshape((-1, 1))], axis=1),
            columns=["x1", "x2", "x3", "y"],
        )

        res = lm(data, "y ~ x1 + x2 + x3", dtype=jnp.float64)
        self.assertAllClose(res(X), y.reshape((-1, 1)), atol=1e-5)
        self.assertAllClose(
            np.sum(np.multiply(res.resid, X), axis=1), np.zeros(1000), atol=1e-4
        )
        self.assertEqual(res.df_model, 3)
        self.assertEqual(res.df_resid, 996)

    def test_olsstate_err_t(self):
        np.random.seed(0)
        X = np.random.uniform(size=(1000, 3))
        y = (
            3 * X[:, 0]
            + 2 * X[:, 1]
            - 7 * X[:, 2]
            + np.random.standard_t(10, size=(1000,))
        )
        data = pd.DataFrame(
            np.concatenate([X, y.reshape((-1, 1))], axis=1),
            columns=["x1", "x2", "x3", "y"],
        )
        res = lm(data, "y ~ x1 + x2 + x3", dtype=jnp.float64)
        res_smf = smf.ols("y ~ x1 + x2 + x3", data=data).fit()
        self.assertAllClose(
            res.coefs.reshape(-1), res_smf.params.values.reshape(-1), atol=1e-5
        )
        self.assertAllClose(res.F_value, res_smf.fvalue, atol=1e-5)
        self.assertAllClose(
            res.StdErr.reshape(-1), res_smf.bse.values.reshape(-1), atol=1e-5
        )
        self.assertAllClose(
            res.t_values.reshape(-1), res_smf.tvalues.values.reshape(-1), atol=1e-5
        )
        self.assertAllClose(res.R2, res_smf.rsquared, atol=1e-5)
        self.assertAllClose(res.R2_adj, res_smf.rsquared_adj, atol=1e-5)
        self.assertAllClose(res.t_pvalues.reshape(-1), res_smf.pvalues, atol=1e-5)

    def test_pipe_olsstate(self):
        X = np.random.uniform(size=(1000, 3))
        y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
        data = pd.DataFrame(
            np.concatenate([X, y.reshape((-1, 1))], axis=1),
            columns=["x1", "x2", "x3", "y"],
        )

        model = (Pipeable(data) >> lm(formula="y ~ x1 + x2 + x3", dtype=jnp.float64))()
        trans_X = (Pipeable(X) >> model)()
        self.assertAllClose(trans_X, y.reshape((-1, 1)), atol=1e-5)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
