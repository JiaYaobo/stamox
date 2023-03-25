"""Test for laplace distribution"""

import jax.random as jrand
import jax.numpy as jnp

import numpy as np

from stamox.distribution import plaplace, rlaplace, qlaplace, dlaplace

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtest



class LaplaceTest(jtest.JaxTestCase):

    def test_rlaplace(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (100000000, )
        laplaces = rlaplace(key, sample_shape=sample_shape)
        mean = laplaces.mean()
        var = laplaces.var(ddof=1)
        self.assertAllClose(mean, 0., atol=1e-2, rtol=1e-4)
        self.assertAllClose(var, 2 * 1.**2, atol=1e-2, rtol=1e-4)

    def test_plaplace(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        loc = 0.
        scale = 1.
        p = plaplace(x, loc, scale)
        true_p = np.array(
            [0.5475813 ,0.5906346 ,0.6295909 ,0.6648400 ,0.6967347])
        self.assertArraysAllClose(p, true_p)

    def test_qlaplace(self):
        q = np.array(
            [0.5475813 ,0.5906346 ,0.6295909 ,0.6648400 ,0.6967347])
        loc = 0.
        scale = 1.
        x = qlaplace(q, loc, scale)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_dlaplace(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        loc = 0.
        scale = 1.
        grads = dlaplace(x, loc, scale)
        true_grads = np.array(
            [0.4524187 ,0.4093654 ,0.3704091,0.3351600 ,0.3032653])
        self.assertArraysAllClose(grads, true_grads)


if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())
