"""Test the ECDF."""
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest

import stamox.pipe_functions as PF
from stamox.distribution import ecdf


class ECDFTest(jtest.JaxTestCase):
    def test_ecdf(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y = ecdf(x)(x)
        true_y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertArraysAllClose(y, true_y)

    def test_pipe_ecdf(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y = PF.ecdf()(x)(x)
        true_y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertArraysAllClose(y, true_y)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
