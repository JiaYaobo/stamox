"""Test for normal distribution"""
from absl.testing import absltest

from jax._src import test_util as jtest
import jax.random as jrand
import numpy as np

from stamox.distribution import pnorm, qnorm, rnorm, dnorm


class NormalTest(jtest.JaxTestCase):
    def test_rnorm(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        mean = 0.5
        sd = 2.0
        norms = rnorm(key, mean=mean, sd=sd, sample_shape=sample_shape)
        avg = norms.mean()
        var = norms.var(ddof=1)
        self.assertAllClose(avg, mean, atol=1e-2)
        self.assertAllClose(var, sd**2, atol=1e-2)

    def test_pnorm(self):
        x = np.array([-1.96, -1.645, -1.0, 0, 1.0, 1.645, 1.96])
        p = pnorm(x)
        true_p = np.array(
            [0.02499789, 0.04998492, 0.15865527, 0.5, 0.8413447, 0.95001507, 0.9750021]
        )
        self.assertArraysAllClose(p, true_p)

    def test_qnorm(self):
        q = np.array(
            [0.02499789, 0.04998492, 0.15865527, 0.5, 0.8413447, 0.95001507, 0.9750021]
        )
        x = qnorm(q)
        true_x = np.array([-1.96, -1.645, -1.0, 0, 1.0, 1.645, 1.96])
        self.assertArraysAllClose(x, true_x)

    def test_dnorm(self):
        x = np.array([-1.96, -1.645, -1.0, 0, 1.0, 1.645, 1.96])
        grads = dnorm(x)
        true_grads = np.array(
            [
                0.05844094,
                0.10311081,
                0.24197072,
                0.39894228,
                0.24197072,
                0.10311081,
                0.05844094,
            ]
        )
        self.assertArraysAllClose(grads, true_grads)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
