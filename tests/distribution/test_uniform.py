"""Test the uniform distribution."""
import jax.random as jrand
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.distribution import dunif, punif, qunif, runif


class UniformTest(jtest.JaxTestCase):

    def test_runif(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        ts = runif(key, sample_shape)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, 1 / 2, atol=1e-2)
        self.assertAllClose(var, 1 / 12, atol=1e-2)
    
    def test_punif(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        p = punif(x)
        true_p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(p, true_p)

    def test_qunif(self):
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        x = qunif(q)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)
    
    def test_dunif(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        grads = dunif(x)
        true_grads = np.array([1., 1., 1., 1., 1.])
        self.assertArraysAllClose(grads, true_grads)
    
    def test_partial_punif(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        p = punif()(x)
        true_p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(p, true_p)
    
    def test_partial_qunif(self):
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        x = qunif()(q)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)
    
    def test_partial_dunif(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        grads = dunif()(x)
        true_grads = np.array([1., 1., 1., 1., 1.])
        self.assertArraysAllClose(grads, true_grads)
    
    def test_partial_runif(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        ts = runif()(key, sample_shape)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, 1 / 2, atol=1e-2)
        self.assertAllClose(var, 1 / 12, atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())