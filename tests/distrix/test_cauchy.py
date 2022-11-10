"""Test for cauchy distribution"""

import jax.random as jrand
import jax.numpy as jnp

import numpy as np

from stamox.distrix import pcauchy, rcauchy, qcauchy, dcauchy

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtest



class CauchyTest(jtest.JaxTestCase):

    def test_rcauchy(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000, )
        cauchys = rcauchy(key, sample_shape=sample_shape)
        median = jnp.median(cauchys)
        self.assertAllClose(median, 0., atol=1e-2)

    def test_pcauchy(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        loc = 0.
        scale = 1.
        p = pcauchy(x, loc, scale)
        true_p = np.array(
            [0.5317255 ,0.5628330 ,0.5927736 ,0.6211189,0.6475836])
        self.assertArraysAllClose(p, true_p)

    def test_qcauchy(self):
        q = np.array(
            [0.5317255 ,0.5628330 ,0.5927736 ,0.6211189,0.6475836])
        loc = 0.
        scale = 1.
        x = qcauchy(q, loc, scale)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_dcauchy(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        loc = 0.
        scale = 1.
        grads = dcauchy(x, loc, scale)
        true_grads = np.array(
            [0.3151583 ,0.3060672 ,0.2920274 ,0.2744051 ,0.2546479])
        self.assertArraysAllClose(grads, true_grads)


if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())
