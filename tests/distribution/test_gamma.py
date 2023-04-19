"""Test for gamma distribution"""
import jax.random as jrand
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import gamma

import stamox.pipe_functions as PF
from stamox.distribution import dgamma, pgamma, qgamma, rgamma


class GammaTest(jtest.JaxTestCase):
    def test_rgamma(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        shape = 2.0
        rate = 2.0
        gammas = rgamma(key, sample_shape, shape, rate)
        avg = gammas.mean()
        var = gammas.var(ddof=1)
        self.assertAllClose(avg, shape / rate, atol=1e-2)
        self.assertAllClose(var, shape / rate**2, atol=1e-2)

    def test_pgamma(self):
        x = np.random.gamma(2, 2, 1000)
        shape = 2.0
        rate = 2.0
        p = pgamma(x, shape, rate)
        true_p = gamma.cdf(x, shape, scale=1 / rate)
        self.assertArraysAllClose(p, true_p)

    def test_qgamma(self):
        q = np.random.uniform(0, 0.999, 1000)
        shape = 2.0
        rate = 2.0
        x = qgamma(q, shape, rate)
        true_x = gamma.ppf(q, shape, scale=1 / rate)
        self.assertArraysAllClose(x, true_x)

    def test_dgamma(self):
        x = np.random.gamma(2, 2, 1000)
        shape = 2.0
        rate = 2.0
        grads = dgamma(x, shape, rate)
        true_grads = gamma.pdf(x, shape, scale=1 / rate)
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_pgamma(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        shape = 2.0
        rate = 2.0
        p = PF.pgamma(shape=shape, rate=rate)(x)
        true_p = np.array([0.01752310, 0.06155194, 0.12190138, 0.19120786, 0.26424112])
        self.assertArraysAllClose(p, true_p)

    def test_partial_qgamma(self):
        q = np.array([0.01752310, 0.06155194, 0.12190138, 0.19120786, 0.26424112])
        shape = 2.0
        rate = 2.0
        x = PF.qgamma(shape=shape, rate=rate)(q)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_partial_dgamma(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        shape = 2.0
        rate = 2.0
        grads = PF.dgamma(shape=shape, rate=rate)(x)
        true_grads = np.array([0.3274923, 0.5362560, 0.6585740, 0.7189263, 0.7357589])
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_rgamma(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        shape = 2.0
        rate = 2.0
        gammas = PF.rgamma(shape=shape, rate=rate, sample_shape=sample_shape)(key)
        avg = gammas.mean()
        var = gammas.var(ddof=1)
        self.assertAllClose(avg, shape / rate, atol=1e-2)
        self.assertAllClose(var, shape / rate**2, atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
