"""Test for F distribution"""
import jax.random as jrand
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import f

from stamox.distribution import dF, pF, qF, rF


np.random.seed(1)


class FTest(jtest.JaxTestCase):
    def test_rF(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        dfn = 5.0
        dfd = 5.0
        rvs = rF(key, sample_shape, dfn, dfd)
        avg = rvs.mean()
        self.assertAllClose(avg, dfd / (dfd - 2), atol=1e-2)

    def test_pF(self):
        x = np.random.f(5, 5, 1000)
        dfn = 5.0
        dfd = 5.0
        p = pF(x, dfn, dfd)
        true_p = f.cdf(x, dfn, dfd)
        self.assertArraysAllClose(p, true_p, atol=1e-5)

    def test_qF(self):
        q = np.random.uniform(0, 1., 1000)
        dfn = 5.0
        dfd = 5.0
        x = qF(q, dfn, dfd)
        true_x = f.ppf(q, dfn, dfd)
        self.assertArraysAllClose(x, true_x, atol=1e-4)

    def test_dF(self):
        x = np.random.f(5, 5, 1000)
        dfn = 5.0
        dfd = 5.0
        grads = dF(x, dfn, dfd)
        true_grads = f.pdf(x, dfn, dfd)
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_pF(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        dfn = 5.0
        dfd = 5.0
        p = pF(dfn=dfn, dfd=dfd)(x)
        true_p = np.array([0.01224192, 0.05096974, 0.10620910, 0.16868416, 0.23251132])
        self.assertArraysAllClose(p, true_p)

    def test_partial_dF(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        dfn = 5.0
        dfd = 5.0
        grads = dF(dfn=dfn, dfd=dfd)(x)
        true_grads = np.array([0.2666708, 0.4881773, 0.6010408, 0.6388349, 0.6323209])
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_qF(self):
        q = np.array([0.01224192, 0.05096974, 0.10620910, 0.16868416, 0.23251132])
        dfn = 5.0
        dfd = 5.0
        x = qF(dfn=dfn, dfd=dfd)(q)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_partial_rF(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        dfn = 5.0
        dfd = 5.0
        rvs = rF(dfn=dfn, dfd=dfd, sample_shape=sample_shape)(key)
        avg = rvs.mean()
        self.assertAllClose(avg, dfd / (dfd - 2), atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
