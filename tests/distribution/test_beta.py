"""Test for beta distribution"""
from absl.testing import absltest

import jax.random as jrand
from jax._src import test_util as jtest
import numpy as np

from stamox.distribution import pbeta, rbeta, qbeta, dbeta


class BetaTest(jtest.JaxTestCase):
    def test_rbeta(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        a = 2.0
        b = 2.0
        ts = rbeta(key, a, b, sample_shape)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, a / (a + b), atol=1e-2)
        self.assertAllClose(var, a * b / (((a + b) ** 2) * (a + b + 1)), atol=1e-2)

    def test_pbeta(self):
        x = np.array([0.0, 1.0, 0.2, 0.3, 0.4, 0.5])
        a = 2.0
        b = 2.0
        p = pbeta(x, a, b)
        true_p = np.array([0.000, 1.000, 0.104, 0.216, 0.352, 0.500])
        self.assertArraysAllClose(p, true_p)

    def test_qbeta(self):
        q = np.array([0.000, 1.000, 0.104, 0.216, 0.352, 0.500])
        x = qbeta(q, 2.0, 2.0)
        true_x = np.array([0.0, 1.0, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_dbeta(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        grads = dbeta(x, 2, 2)
        true_grads = np.array([0.54, 0.96, 1.26, 1.44, 1.50])
        self.assertArraysAllClose(grads, true_grads)
     
    def test_partial_pbeta(self):
        x = np.array([0.0, 1.0, 0.2, 0.3, 0.4, 0.5])
        a = 2.0
        b = 2.0
        p = pbeta(a=a, b=b, lower_tail=False)(x)
        true_p = np.array([1.000, 0.000, 0.896, 0.784, 0.648, 0.500])
        self.assertArraysAllClose(p, true_p)
    
    def test_partial_qbeta(self):
        q = np.array([0.000, 1.000, 0.104, 0.216, 0.352, 0.500])
        x = qbeta(a=2.0, b=2.0, lower_tail=False)(q)
        true_x = np.array([1.0, 0.0, 0.8, 0.7, 0.6, 0.5])
        self.assertArraysAllClose(x, true_x)
    
    def test_partial_dbeta(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        grads = dbeta(a=2, b=2)(x)
        true_grads = np.array([0.54, 0.96, 1.26, 1.44, 1.50])
        self.assertArraysAllClose(grads, true_grads)
    
    def test_partial_rbeta(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        a = 2.0
        b = 2.0
        ts = rbeta(a=a, b=b, sample_shape=sample_shape)(key)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, a / (a + b), atol=1e-2)
        self.assertAllClose(var, a * b / (((a + b) ** 2) * (a + b + 1)), atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
