"""Test for F distribution"""

import jax.random as jrand
import jax.numpy as jnp

import numpy as np

from stamox.distribution import pF, rF, qF, dF

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtest



class FTest(jtest.JaxTestCase):

    def test_rF(self):
        key = jrand.PRNGKey(19751002)
        sample_shape = (1000000, )
        dfn = 5.
        dfd = 5.
        Fs = rF(key, dfn, dfd, sample_shape)
        avg = Fs.mean()
        self.assertAllClose(avg, dfd / (dfd - 2), atol=1e-2)

    def test_pF(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        dfn = 5.
        dfd = 5.
        p = pF(x, dfn, dfd)
        true_p = np.array(
            [0.01224192 ,0.05096974 ,0.10620910 ,0.16868416 ,0.23251132])
        self.assertArraysAllClose(p, true_p)

    def test_qF(self):
        q = np.array(
            [0.01224192 ,0.05096974 ,0.10620910 ,0.16868416 ,0.23251132])
        dfn = 5.
        dfd = 5.
        x = qF(q, dfn, dfd)
        true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertArraysAllClose(x, true_x)

    def test_dF(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        dfn = 5.
        dfd = 5.
        grads = dF(x, dfn, dfd)
        true_grads = np.array(
            [0.2666708 ,0.4881773 ,0.6010408 ,0.6388349 ,0.6323209])
        self.assertArraysAllClose(grads, true_grads)


if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())
