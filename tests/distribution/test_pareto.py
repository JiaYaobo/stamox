"""Test for pareto distribution"""
import jax.random as jrand
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest

import stamox.pipe_functions as PF
from stamox.distribution import dpareto, ppareto, qpareto, rpareto


class ParetoTest(jtest.JaxTestCase):
    def test_rpareto(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        scale = 0.1
        alpha = 3.0
        ts = rpareto(key, sample_shape, scale, alpha)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, alpha * scale / (alpha - 1), atol=1e-2)
        self.assertAllClose(
            var, scale**2 * alpha / (((alpha - 1) ** 2) * (alpha - 2)), atol=1e-2
        )

    def test_ppareto(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        scale = 0.1
        alpha = 2.0
        p = ppareto(x, scale, alpha)
        true_p = np.array([0.0000000, 0.7500000, 0.8888889, 0.9375000, 0.9600000])
        self.assertArraysAllClose(p, true_p)

    def test_qpareto(self):
        q = np.array([0.0000000, 0.7500000, 0.8888889, 0.9375000, 0.9600000])
        x = qpareto(q, 0.1, 2.0)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_dpareto(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        grads = dpareto(x, 0.1, 2.0)
        true_grads = np.array([20.0000000, 2.5000000, 0.7407407, 0.3125000, 0.1600000])
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_ppareto(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        scale = 0.1
        alpha = 2.0
        p = PF.ppareto(scale=scale, alpha=alpha)(x)
        true_p = np.array([0.0000000, 0.7500000, 0.8888889, 0.9375000, 0.9600000])
        self.assertArraysAllClose(p, true_p)

    def test_partial_qpareto(self):
        q = np.array([0.0000000, 0.7500000, 0.8888889, 0.9375000, 0.9600000])
        x = PF.qpareto(scale=0.1, alpha=2.0)(q)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_partial_dpareto(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        grads = PF.dpareto(scale=0.1, alpha=2.0)(x)
        true_grads = np.array([20.0000000, 2.5000000, 0.7407407, 0.3125000, 0.1600000])
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_rpareto(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        scale = 0.1
        alpha = 3.0
        ts = PF.rpareto(scale=scale, alpha=alpha, sample_shape=sample_shape)(key)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, alpha * scale / (alpha - 1), atol=1e-2)
        self.assertAllClose(
            var, scale**2 * alpha / (((alpha - 1) ** 2) * (alpha - 2)), atol=1e-2
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
