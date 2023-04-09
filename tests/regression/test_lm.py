"""Test Linear Regression"""
import numpy as np
import pandas as pd
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.core import Pipeable
from stamox.regression import lm


class LMTest(jtest.JaxTestCase):
    def test_lm_fullset(self):
        X = np.random.uniform(size=(1000, 3))
        y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
        data = pd.DataFrame(
            np.concatenate([X, y.reshape((-1, 1))], axis=1),
            columns=["x1", "x2", "x3", "y"],
        )

        res = lm(data, "y ~ x1 + x2 + x3")
        self.assertAllClose(
            res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
        )
        self.assertEqual(res.df_resid, 996)
        self.assertEqual(res.df_model, 3)

    def test_lm_subset(self):
        subset = [i for i in range(500)]
        X = np.random.uniform(size=(1000, 3))
        y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
        data = pd.DataFrame(
            np.concatenate([X, y.reshape((-1, 1))], axis=1),
            columns=["x1", "x2", "x3", "y"],
        )

        res = lm(data, "y ~ x1 + x2 + x3", subset=subset)
        self.assertAllClose(
            res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
        )
        self.assertEqual(res.df_resid, 496)
        self.assertEqual(res.df_model, 3)

    def test_lm_weighted(self):
        weights = np.abs(np.random.uniform(size=(1000, 1)))
        X = np.random.uniform(size=(1000, 3))
        y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
        data = pd.DataFrame(
            np.concatenate([X, y.reshape((-1, 1))], axis=1),
            columns=["x1", "x2", "x3", "y"],
        )

        res = lm(data, "y ~ x1 + x2 + x3", weights=weights)
        self.assertAllClose(
            res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
        )
        self.assertEqual(res.df_resid, 996)
        self.assertEqual(res.df_model, 3)

    def test_lm_svd(self):
        X = np.random.uniform(size=(1000, 3))
        y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
        data = pd.DataFrame(
            np.concatenate([X, y.reshape((-1, 1))], axis=1),
            columns=["x1", "x2", "x3", "y"],
        )

        res = lm(data, "y ~ x1 + x2 + x3", method="svd")
        self.assertAllClose(
            res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
        )

    def test_lm_inv(self):
        X = np.random.uniform(size=(1000, 3))
        y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
        data = pd.DataFrame(
            np.concatenate([X, y.reshape((-1, 1))], axis=1),
            columns=["x1", "x2", "x3", "y"],
        )

        res = lm(data, "y ~ x1 + x2 + x3", method="inv")
        self.assertAllClose(
            res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
        )

    def test_lm_with_NA(self):
        X = np.random.uniform(size=(1000, 3))
        y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
        data = pd.DataFrame(
            np.concatenate([X, y.reshape((-1, 1))], axis=1),
            columns=["x1", "x2", "x3", "y"],
        )
        data["x1"][0] = np.nan
        data["x2"][1] = np.nan
        data["x3"][2] = np.nan
        data["y"][3] = np.nan

        res = lm(data, "y ~ x1 + x2 + x3")
        self.assertAllClose(
            res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
        )

    def test_pipe_lm(self):
        X = np.random.uniform(size=(1000, 3))
        y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
        data = pd.DataFrame(
            np.concatenate([X, y.reshape((-1, 1))], axis=1),
            columns=["x1", "x2", "x3", "y"],
        )

        res = (Pipeable(data) >> lm(formula="y ~ x1 + x2 + x3"))()
        self.assertAllClose(
            res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
        )

    def test_lm_list_input(self):
        X = np.random.uniform(size=(1000, 3))
        X = np.concatenate([np.ones((1000, 1)), X], axis=1)
        y = 3 * X[:, 1] + 2 * X[:, 2] - 7 * X[:, 3] + 1.0
        data = [y, X]
        res = lm(data)
        self.assertAllClose(
            res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
        )
    
    def test_lm_tuple_input(self):
        X = np.random.uniform(size=(1000, 3))
        X = np.concatenate([np.ones((1000, 1)), X], axis=1)
        y = 3 * X[:, 1] + 2 * X[:, 2] - 7 * X[:, 3] + 1.0
        data = (y, X)
        res = lm(data)
        self.assertAllClose(
            res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
        )
    
    def test_lm_ndarray_input(self):
        X = np.random.uniform(size=(1000, 3))
        X = np.concatenate([np.ones((1000, 1)), X], axis=1)
        y = 3 * X[:, 1] + 2 * X[:, 2] - 7 * X[:, 3] + 1.0
        data = np.concatenate([y.reshape((-1, 1)), X], axis=1)
        res = lm(data)
        self.assertAllClose(
            res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
