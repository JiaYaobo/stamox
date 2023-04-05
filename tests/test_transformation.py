"""Test the transformation module."""
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import boxcox as scp_boxcox

from stamox.core import Pipeable
from stamox.transformation import boxcox


class TransformationTest(jtest.JaxTestCase):

    def test_boxcox(self):
        x = np.random.gamma(2, 2, size=(10000, 3))
        lmbda = 2.
        self.assertAllClose(boxcox(x, lmbda), scp_boxcox(x, lmbda))
    
    def test_partial_pipe_boxcox(self):
        x = np.random.gamma(2, 2, size=(10000, 3))
        lmbda = 2.
        p = Pipeable(x) >> boxcox(lmbda=2.)
        self.assertAllClose(p(x), scp_boxcox(x, lmbda))


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())