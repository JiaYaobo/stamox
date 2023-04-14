"""Test for exponential distribution"""
import jax.random as jrand
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import expon

from stamox.distribution import dexp, pexp, qexp, rexp


np.random.seed(1)


class ExpTest(jtest.JaxTestCase):
    def test_rexp(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        rate = 2.5
        rvs = rexp(key, sample_shape, rate)
        np_rvs = np.random.exponential(1 / rate, sample_shape)
        np_avg = np_rvs.mean()
        np_var = np_rvs.var(ddof=1)
        avg = rvs.mean()
        var = rvs.var(ddof=1)
        self.assertAllClose(avg, np_avg, atol=1e-2)
        self.assertAllClose(var, np_var, atol=1e-2)

    def test_pexp(self):
        x = np.random.exponential(1 / 2.5, 1000)
        rate = 2.5
        p = pexp(x, rate)
        true_p = expon(scale=1 / rate).cdf(x)
        self.assertArraysAllClose(p, true_p)

    def test_qexp(self):
        q = np.random.uniform(0, 0.999, 1000)
        x = qexp(q, 2.5)
        true_x = expon(scale=1 / 2.5).ppf(q)
        self.assertArraysAllClose(x, true_x)

    def test_dexp(self):
        x = np.random.exponential(1 / 2.5, 1000)
        rate = 2.5
        grads = dexp(x, rate)
        true_grads = np.exp(-x * rate) * rate
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_pexp(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rate = 2.5
        p = pexp(rate=rate)(x)
        true_p = np.array([0.2211992, 0.3934693, 0.5276334, 0.6321206, 0.7134952])
        self.assertArraysAllClose(p, true_p)

    def test_partial_qexp(self):
        q = np.array([0.2211992, 0.3934693, 0.5276334, 0.6321206, 0.7134952])
        x = qexp(rate=2.5)(q)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_partial_dexp(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rate = 2.5
        grads = dexp(rate=rate)(x)
        true_grads = np.array([1.9470020, 1.5163266, 1.1809164, 0.9196986, 0.7162620])
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_rexp(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        rate = 2.5
        rvs = rexp(rate=rate, sample_shape=sample_shape)(key)
        avg = rvs.mean()
        var = rvs.var(ddof=1)
        self.assertAllClose(avg, 1 / rate, atol=1e-2)
        self.assertAllClose(var, 1 / rate**2, atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
