"""Test for student t distribution"""
import jax.random as jrand
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import t

from stamox.distribution import dt, pt, qt, rt


np.random.seed(19751002)


class TTest(jtest.JaxTestCase):
    def test_rt(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        df = 6
        ts = rt(key, sample_shape, df, 0.0, 1.0)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, 0.0, atol=1e-2)
        self.assertAllClose(var, float(df / (df - 2)), atol=1e-2)

    def test_pt(self):
        x = np.random.standard_t(3, 10000)
        p = pt(x, 3)
        true_p = t.cdf(x, 3).astype(np.float32)
        self.assertArraysAllClose(p, true_p, atol=1e-4)

    def test_qt(self):
        q = np.random.uniform(0, 0.999, 10000)
        x = qt(q, 6)
        true_x = t.ppf(q, 6).astype(np.float32)
        self.assertArraysAllClose(x, true_x, atol=1e-3)

    def test_dt(self):
        x = np.random.standard_t(6, 10)
        grads = dt(x, 6)
        true_grads = t.pdf(x, 6).astype(np.float32)
        self.assertArraysAllClose(grads, true_grads, atol=1e-4)

    def test_partial_pt(self):
        x = np.array([1.0, 1.645, 1.96, 2.65, 3.74])
        p = pt(df=6)(x)
        true_p = np.array([0.8220412, 0.9244638, 0.9511524, 0.9809857, 0.9951888])
        self.assertArraysAllClose(p, true_p)

    def test_partial_qt(self):
        q = np.array([0.8220412, 0.9244638, 0.9511524, 0.9809857, 0.9951888])
        x = qt(df=6)(q)
        true_x = np.array([1.0, 1.645, 1.96, 2.65, 3.74])
        self.assertArraysAllClose(x, true_x)

    def test_partial_dt(self):
        x = np.array([1.0, 1.645, 1.96, 2.65, 3.74])
        grads = dt(df=6)(x)
        true_grads = np.array(
            [0.223142291, 0.104005269, 0.067716584, 0.025409420, 0.005672347]
        )
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_rt(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000,)
        rvs = rt(df=6, loc=0.0, scale=1.0, sample_shape=sample_shape)(key)
        avg = rvs.mean()
        var = rvs.var(ddof=1)
        self.assertAllClose(avg, 0.0, atol=1e-2)
        self.assertAllClose(var, float(6 / (6 - 2)), atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
