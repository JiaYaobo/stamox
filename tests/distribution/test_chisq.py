"""Test for chisqonential distribution"""

import jax.random as jrand

import numpy as np

from stamox.distribution import pchisq, rchisq, qchisq, dchisq

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtest


class ChisqTest(jtest.JaxTestCase):

    def test_rchisq(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (10000000, )
        df = 3.
        ts = rchisq(key, df, sample_shape)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, df, atol=1e-2)
        self.assertAllClose(var, 2 * df, atol=1e-2)

    def test_pchisq(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        df = 3
        p = pchisq(x, df)
        true_p = np.array(
            [0.008162576, 0.022410702, 0.039971520, 0.059757505, 0.081108588])
        self.assertArraysAllClose(p, true_p)

    def test_qchisq(self):
        q = np.array([0.008162576, 0.022410702,
                     0.039971520, 0.059757505, 0.081108588])
        df = 3
        x = qchisq(q, df)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_dchisq(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        df = 3.
        grads = dchisq(x, df)
        true_grads = np.array(
            [0.1200039, 0.1614342, 0.1880730, 0.2065766, 0.2196956])
        self.assertArraysAllClose(grads, true_grads)


if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())
