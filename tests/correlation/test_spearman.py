"""Test Spearman correlation coefficient"""
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import spearmanr as scp_spearman

from stamox.correlation import spearmanr


class SpearmanRTest(jtest.JaxTestCase):
    def test_corr_1d(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 6, 7, 8, 7]
        z = np.array([x, y], dtype=np.float32)
        self.assertAlmostEqual(scp_spearman(x, y).statistic, spearmanr(x, y))
        self.assertAlmostEqual(scp_spearman(z, axis=1).statistic, spearmanr(z, axis=1))

    def test_corr_2d(self):
        rng = np.random.default_rng()
        x2n = rng.standard_normal((100, 2))
        y2n = rng.standard_normal((100, 2))
        self.assertAlmostEqual(scp_spearman(x2n).statistic, spearmanr(x2n))
        self.assertAlmostEqual(scp_spearman(y2n).statistic, spearmanr(y2n))
        self.assertAllClose(scp_spearman(x2n, y2n).statistic, spearmanr(x2n, y2n))
        self.assertAllClose(scp_spearman(x2n, y2n).statistic, spearmanr(x2n, y2n))
        self.assertAllClose(
            scp_spearman(x2n, y2n, axis=None).statistic, spearmanr(x2n, y2n, axis=None)
        )
    
    def test_pipe(self):
        rng = np.random.default_rng()
        x2n = rng.standard_normal((100, 2))
        y2n = rng.standard_normal((100, 2))
        f = spearmanr(y=y2n[:, 0], axis=0)
        g = spearmanr(y=x2n[:, 0], axis=0)
        self.assertAllClose(f(y2n[:, 1]), spearmanr(y2n, axis=0))
        self.assertAllClose(g(x2n[:, 1]), spearmanr(x2n, axis=0))


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
