"""Test for chisqonential distribution"""

import jax.random as jrand
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import chi2

from stamox.distribution import dchisq, pchisq, qchisq, rchisq


np.random.seed(1)
class ChisqTest(jtest.JaxTestCase):

    def test_rchisq(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (10000000, )
        df = 3.
        ts = rchisq(key, sample_shape, df)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, df, atol=1e-2)
        self.assertAllClose(var, 2 * df, atol=1e-2)

    def test_pchisq(self):
        x = np.random.chisquare(3, 1000)
        df = 3
        p = pchisq(x, df)
        true_p = chi2(df).cdf(x)
        self.assertArraysAllClose(p, true_p)

    def test_qchisq(self):
        q = np.random.uniform(0, 0.999, 1000)
        df = 3
        x = qchisq(q, df)
        true_x = chi2(df).ppf(q)
        self.assertArraysAllClose(x, true_x)

    def test_dchisq(self):
        x = np.random.chisquare(3, 1000)
        df = 3.
        grads = dchisq(x, df)
        true_grads = chi2(df).pdf(x)
        self.assertArraysAllClose(grads, true_grads)
    
    def test_partial_pchisq(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        df = 3
        p = pchisq(df=df, lower_tail=False)(x)
        true_p = np.array(
            [0.991837424, 0.977589298, 0.960028480, 0.940242495, 0.918891412])
        self.assertArraysAllClose(p, true_p)
    
    def test_partial_qchisq(self):
        q = 1 - np.array([0.008162576, 0.022410702,
                     0.039971520, 0.059757505, 0.081108588])
        df = 3
        x = qchisq(df=df, lower_tail=False)(q)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)
    
    def test_partial_dchisq(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        df = 3.
        grads = dchisq(df=df)(x)
        true_grads = np.array(
            [0.1200039, 0.1614342, 0.1880730, 0.2065766, 0.2196956])
        self.assertArraysAllClose(grads, true_grads)

    def test_partial_rchisq(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (10000000, )
        df = 3.
        ts = rchisq(df=df)(key, sample_shape=sample_shape) 
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, df, atol=1e-2)
        self.assertAllClose(var, 2 * df, atol=1e-2)


if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())
