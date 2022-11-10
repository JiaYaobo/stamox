"""Test for weibull distribution"""

import jax.random as jrand
import jax.numpy as jnp

import numpy as np

from stamox.distrix import pweibull, rweibull, qweibull, dweibull

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtest


class WeiBullTest(jtest.JaxTestCase):

    def test_rweibull(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000, )
        a = 2.
        b = 2.
        ts = rweibull(key, a, b, sample_shape)
        avg = ts.mean()
        var = ts.var(ddof=1)
        self.assertAllClose(avg, a / (a + b), atol=1e-2)
        self.assertAllClose(var, a*b / (((a+b)**2)*(a+b+1)), atol=1e-2)

    def test_pweibull(self):
        x = np.array([0., 1., 0.2, 0.3, 0.4, 0.5])
        a = 2.
        b = 2.
        p = pweibull(x, a, b)
        true_p = np.array([0.000, 1.000, 0.104, 0.216, 0.352, 0.500])
        self.assertArraysAllClose(p, true_p)

    def test_qweibull(self):
        q = np.array([0.000, 1.000, 0.104, 0.216, 0.352, 0.500])
        x = qweibull(q, 2, 2)
        true_x = np.array([0., 1., 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_dweibull(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        grads = dweibull(x, 2, 2)
        true_grads = np.array(
            [0.54, 0.96, 1.26, 1.44, 1.50])
        self.assertArraysAllClose(grads, true_grads)


if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())
