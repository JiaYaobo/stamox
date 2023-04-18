"""Test the transformation module."""
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import boxcox as scp_boxcox

from stamox.core import Pipeable
from stamox.transformation import boxcox, z_fisher


class TransformationTest(jtest.JaxTestCase):
    def test_boxcox(self):
        x = np.random.gamma(2, 2, size=(10000, 3))
        lmbda = 2.0
        self.assertAllClose(boxcox(x, lmbda), scp_boxcox(x, lmbda))

    def test_partial_pipe_boxcox(self):
        x = np.random.gamma(2, 2, size=(10000, 3))
        lmbda = 2.0
        p = Pipeable(x) >> boxcox(lmbda=2.0)
        self.assertAllClose(p(x), scp_boxcox(x, lmbda))

    def test_z_fisher(self):
        x = np.random.uniform(size=1000)
        self.assertAllClose(z_fisher(x), np.arctanh(x), atol=1e-4)

    def test_partial_pipe_z_fisher(self):
        x = np.random.uniform(size=1000)
        p = Pipeable(x) >> z_fisher
        self.assertAllClose(p(x), np.arctanh(x), atol=1e-4)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
