"""Test for weibull distribution"""
from absl.testing import absltest

from jax._src import test_util as jtest

import jax.random as jrand
import numpy as np

from stamox.distribution import pweibull, rweibull, qweibull, dweibull


class WeiBullTest(jtest.JaxTestCase):
    def test_rweibull(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        a = 1.0
        b = 1.0
        ts = rweibull(key, a, b, sample_shape)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, 1.0, atol=1e-2)
        self.assertAllClose(var, 1.0, atol=1e-2)

    def test_pweibull(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        a = 1.0
        b = 1.0
        p = pweibull(x, a, b)
        true_p = np.array([0.09516258, 0.18126925, 0.25918178, 0.32967995, 0.39346934])
        self.assertArraysAllClose(p, true_p)

    def test_qweibull(self):
        q = np.array([0.09516258, 0.18126925, 0.25918178, 0.32967995, 0.39346934])
        x = qweibull(q, 1.0, 1.0)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_dweibull(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        grads = dweibull(x, 1.0, 1.0)
        true_grads = np.array([0.9048374, 0.8187308, 0.7408182, 0.6703200, 0.6065307])
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_pweibull(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        a = 1.0
        b = 1.0
        p = pweibull(concentration=a, scale=b)(x)
        true_p = np.array([0.09516258, 0.18126925, 0.25918178, 0.32967995, 0.39346934])
        self.assertArraysAllClose(p, true_p)

    def test_partial_qweibull(self):
        q = np.array([0.09516258, 0.18126925, 0.25918178, 0.32967995, 0.39346934])
        a = 1.0
        b = 1.0
        x = qweibull(concentration=a, scale=b)(q)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_partial_dweibull(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        a = 1.0
        b = 1.0
        grads = dweibull(concentration=a, scale=b)(x)
        true_grads = np.array([0.9048374, 0.8187308, 0.7408182, 0.6703200, 0.6065307])
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_rweibull(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        a = 1.0
        b = 1.0
        ts = rweibull(concentration=a, scale=b, sample_shape=sample_shape)(key)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, 1.0, atol=1e-2)
        self.assertAllClose(var, 1.0, atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
