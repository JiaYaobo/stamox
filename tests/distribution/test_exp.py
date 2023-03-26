"""Test for exponential distribution"""
from absl.testing import absltest
from jax._src import test_util as jtest
import jax.random as jrand
import numpy as np

from stamox.distribution import pexp, rexp, qexp, dexp


class ExpTest(jtest.JaxTestCase):
    def test_rexp(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        rate = 2.5
        ts = rexp(key, rate, sample_shape)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, 1 / rate, atol=1e-2)
        self.assertAllClose(var, 1 / rate**2, atol=1e-2)

    def test_pexp(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rate = 2.5
        p = pexp(x, rate)
        true_p = np.array([0.2211992, 0.3934693, 0.5276334, 0.6321206, 0.7134952])
        self.assertArraysAllClose(p, true_p)

    def test_qexp(self):
        q = np.array([0.2211992, 0.3934693, 0.5276334, 0.6321206, 0.7134952])
        x = qexp(q, 2.5)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_dexp(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rate = 2.5
        grads = dexp(x, rate)
        true_grads = np.array([1.9470020, 1.5163266, 1.1809164, 0.9196986, 0.7162620])
        self.assertArraysAllClose(grads, true_grads)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
