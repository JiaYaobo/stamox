"""Test for gamma distribution"""
from absl.testing import absltest

import jax.random as jrand
from jax._src import test_util as jtest
import numpy as np

from stamox.distribution import pgamma, rgamma, qgamma, dgamma


class GammaTest(jtest.JaxTestCase):
    def test_rgamma(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        shape = 2.0
        rate = 2.0
        gammas = rgamma(key, shape, rate, sample_shape)
        avg = gammas.mean()
        var = gammas.var(ddof=1)
        self.assertAllClose(avg, shape / rate, atol=1e-2)
        self.assertAllClose(var, shape / rate**2, atol=1e-2)

    def test_pgamma(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        shape = 2.0
        rate = 2.0
        p = pgamma(x, shape, rate)
        true_p = np.array([0.01752310, 0.06155194, 0.12190138, 0.19120786, 0.26424112])
        self.assertArraysAllClose(p, true_p)

    def test_qgamma(self):
        q = np.array([0.01752310, 0.06155194, 0.12190138, 0.19120786, 0.26424112])
        shape = 2.0
        rate = 2.0
        x = qgamma(q, shape, rate)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_dgamma(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        shape = 2.0
        rate = 2.0
        grads = dgamma(x, shape, rate)
        true_grads = np.array([0.3274923, 0.5362560, 0.6585740, 0.7189263, 0.7357589])
        self.assertArraysAllClose(grads, true_grads)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
