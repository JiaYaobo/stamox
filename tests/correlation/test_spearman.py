"""Test Spearman correlation coefficient"""
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import spearmanr as scp_spearman

import stamox.pipe_functions as PF
from stamox.correlation import spearmanr


def test_corr_1d():
    x = [1, 2, 3, 4, 5]
    y = [5, 6, 7, 8, 7]
    z = np.array([x, y], dtype=np.float32)
    np.testing.assert_almost_equal(scp_spearman(x, y).statistic, spearmanr(x, y))
    np.testing.assert_almost_equal(
        scp_spearman(z, axis=1).statistic, spearmanr(z, axis=1)
    )


def test_corr_2d():
    rng = np.random.default_rng()
    x2n = rng.standard_normal((100, 2))
    y2n = rng.standard_normal((100, 2))
    np.testing.assert_almost_equal(scp_spearman(x2n).statistic, spearmanr(x2n))
    np.testing.assert_almost_equal(scp_spearman(y2n).statistic, spearmanr(y2n))
    np.testing.assert_allclose(scp_spearman(x2n, y2n).statistic, spearmanr(x2n, y2n))
    np.testing.assert_allclose(scp_spearman(x2n, y2n).statistic, spearmanr(x2n, y2n))
    np.testing.assert_allclose(
        scp_spearman(x2n, y2n, axis=None).statistic, spearmanr(x2n, y2n, axis=None)
    )


def test_pipe():
    rng = np.random.default_rng()
    x2n = rng.standard_normal((100, 2))
    y2n = rng.standard_normal((100, 2))
    f = PF.spearmanr(y=y2n[:, 0], axis=0)
    g = PF.spearmanr(y=x2n[:, 0], axis=0)
    np.testing.assert_allclose(f(y2n[:, 1]), spearmanr(y2n, axis=0))
    np.testing.assert_allclose(g(x2n[:, 1]), spearmanr(x2n, axis=0))
