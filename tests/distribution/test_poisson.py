"""Test for poisson distribution"""

import jax.random as jrand
import jax.numpy as jnp

import numpy as np

from stamox.distribution import ppoisson, rpoisson, qpoisson, dpoisson

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtest


class PoissonTest(jtest.JaxTestCase):

    def test_rpoisson(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000, )
        rate = 2.5
        ts = rpoisson(key, rate, sample_shape)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, rate, atol=1e-2)
        self.assertAllClose(var, rate, atol=1e-2)

    def test_ppoisson(self):
        x = np.array([1., 2., 3., 4. ,5.])
        rate = 2.5
        p = ppoisson(x, rate)
        true_p = np.array(
            [0.2872975, 0.5438131, 0.7575761, 0.8911780, 0.9579790])
        self.assertArraysAllClose(p, true_p)

    def test_qpoisson(self):
        q = np.array([0.4060058, 0.6766764, 0.8571235, 0.9473470, 0.9834364])
        x = qpoisson(q, 2.)
        true_x = np.array([1., 3., 5., 5., 3.])
        self.assertArraysAllClose(x, true_x)

    def test_dpoisson(self):
        x = np.array([1., 2. , 3., 4., 5.])
        rate = 2.5
        grads = dpoisson(x, rate)
        true_grads = np.array(
            [0.20521250 ,0.25651562 ,0.21376302 ,0.13360189 ,0.06680094])
        self.assertArraysAllClose(grads, true_grads)


if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())
