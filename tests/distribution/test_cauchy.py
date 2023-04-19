"""Test for Cauchy Distribution"""
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
from absl.testing import absltest
from jax import config
from jax._src import test_util as jtest
from scipy.stats import cauchy

import stamox.pipe_functions as PF
from stamox.distribution import dcauchy, pcauchy, qcauchy, rcauchy


config.update("jax_enable_x64", True)
np.random.seed(0)


class CauchyTest(jtest.JaxTestCase):
    def test_rcauchy(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        cauchys = rcauchy(key, sample_shape=sample_shape, dtype=jnp.float64)
        median = jnp.median(cauchys)
        self.assertAllClose(median, 0.0, atol=1e-2)

    def test_pcauchy(self):
        x = np.random.uniform(0, 1, 10000)
        loc = 0.0
        scale = 1.0
        p = pcauchy(x, loc, scale, dtype=jnp.float64)
        true_p = cauchy.cdf(x, loc, scale)
        self.assertArraysAllClose(p, true_p, atol=1e-12)

    def test_qcauchy(self):
        q = np.random.uniform(0, 1, 10000)
        loc = 0.0
        scale = 1.0
        x = qcauchy(q, loc, scale, dtype=jnp.float64)
        true_x = cauchy.ppf(q, loc, scale)
        self.assertArraysAllClose(x, true_x, atol=1e-9)

    def test_dcauchy(self):
        x = np.random.uniform(0, 1, 10000)
        loc = 0.0
        scale = 1.0
        grads = dcauchy(x, loc, scale, dtype=jnp.float64)
        true_grads = cauchy.pdf(x, loc, scale)
        self.assertArraysAllClose(grads, true_grads, atol=1e-12)

    def test_partial_pcauchy(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        loc = 0.0
        scale = 1.0
        p = PF.pcauchy(loc=loc, scale=scale, dtype=jnp.float64)(x)
        true_p = cauchy.cdf(x, loc, scale)
        self.assertArraysAllClose(p, true_p)

    def test_partial_qcauchy(self):
        q = np.array([0.5317255, 0.5628330, 0.5927736, 0.6211189, 0.6475836])
        loc = 0.0
        scale = 1.0
        x = PF.qcauchy(loc=loc, scale=scale, dtype=jnp.float64)(q)
        true_x = cauchy.ppf(q, loc, scale)
        self.assertArraysAllClose(x, true_x)

    def test_partial_dcauchy(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        loc = 0.0
        scale = 1.0
        grads = PF.dcauchy(loc=loc, scale=scale, dtype=jnp.float64)(x)
        true_grads = cauchy.pdf(x, loc, scale)
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_rcauchy(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        cauchys = PF.rcauchy(loc=0.0, scale=1.0, dtype=jnp.float64)(
            key, sample_shape=sample_shape
        )
        median = jnp.median(cauchys)
        self.assertAllClose(median, 0.0, atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
