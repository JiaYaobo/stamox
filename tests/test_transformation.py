"""Test the transformation module."""
import numpy as np
from scipy.stats import boxcox as scp_boxcox

import stamox.pipe_functions as PF
from stamox import Pipeable
from stamox.transformation import boxcox, z_fisher


np.random.seed(0)


def test_boxcox():
    x = np.random.gamma(2, 2, size=(10000, 3))
    lmbda = 2.0
    np.testing.assert_allclose(boxcox(x, lmbda), scp_boxcox(x, lmbda))


def test_partial_pipe_boxcox():
    x = np.random.gamma(2, 2, size=(10000, 3))
    lmbda = 2.0
    p = Pipeable(x) >> PF.boxcox(lmbda=2.0)
    np.testing.assert_allclose(p(x), scp_boxcox(x, lmbda))


def test_z_fisher():
    x = np.random.uniform(size=1000)
    np.testing.assert_allclose(z_fisher(x), np.arctanh(x))


def test_partial_pipe_z_fisher():
    x = np.random.uniform(size=1000)
    p = Pipeable(x) >> PF.z_fisher
    np.testing.assert_allclose(p(x), np.arctanh(x))
