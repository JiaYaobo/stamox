"""Test for OLS"""
import jax.random as jrandom
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.core import Pipeable
from stamox.regression import OLS


class OLSTest(jtest.JaxTestCase):
    def test_ols_with_intercept(self):
        key = jrandom.PRNGKey(0)
        X = jrandom.uniform(key, shape=(1000, 2))
        y = 3 * X[:, 0] + 2 * X[:, 1]  + 1.
        y = y.reshape((-1,1))
        ols = OLS(key=key)
        state = ols.fit(X, y)
        self.assertAllClose(state.params[0].ravel(), np.array([3., 2.]) , atol=1e-5)
        self.assertAllClose(state.params[1], np.array([1.]),  atol=1e-5)
    
    def test_ols_without_intercept(self):
        key = jrandom.PRNGKey(0)
        X = jrandom.uniform(key, shape=(1000, 2))
        y = 3 * X[:, 0] + 2 * X[:, 1]
        y = y.reshape((-1,1))
        ols = OLS(use_intercept=False, key=key)
        state = ols.fit(X, y)
        self.assertAllClose(state.params.ravel(), np.array([3., 2.]) , atol=1e-5)

    def test_pipe_ols_with_intercept(self):
        key = jrandom.PRNGKey(0)
        X = jrandom.uniform(key, shape=(1000, 2))
        y = 3 * X[:, 0] + 2 * X[:, 1]  + 1.
        y = y.reshape((-1,1))
        ols = OLS(key=key)
        h = Pipeable(X, y) >> ols
        state = h()
        self.assertAllClose(state.params[0].ravel(), np.array([3., 2.]) , atol=1e-5)
        self.assertAllClose(state.params[1], np.array([1.]),  atol=1e-5)
    
    def test_pipe_ols_without_intercept(self):
        key = jrandom.PRNGKey(0)
        X = jrandom.uniform(key, shape=(1000, 2))
        y = 3 * X[:, 0] + 2 * X[:, 1]
        y = y.reshape((-1,1))
        ols = OLS(use_intercept=False, key=key)
        h = Pipeable(X, y) >> ols
        state = h()
        self.assertAllClose(state.params.ravel(), np.array([3., 2.]) , atol=1e-5)

if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())