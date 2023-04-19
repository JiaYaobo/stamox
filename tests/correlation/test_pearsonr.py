"""Test Pearson correlation coefficient"""
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import pearsonr as scp_pearsonr

import stamox.pipe_functions as PF
from stamox.correlation import pearsonr


class PearsonRTest(jtest.JaxTestCase):
    def test_correlation_1d(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], np.float32)
        y = np.array([1, 4, 3, 6, 5, 6, 8, 8, 9], np.float32)
        self.assertAllClose(scp_pearsonr(x, x).statistic, pearsonr(x, x))
        self.assertAllClose(scp_pearsonr(x, y).statistic, pearsonr(x, y))
        self.assertAllClose(scp_pearsonr(x, y).statistic, pearsonr([x, y], axis=1))

    def test_correlation_2d(self):
        rng = np.random.default_rng()
        x2n = rng.standard_normal((100, 2))
        y2n = rng.standard_normal((100, 2))
        self.assertAllClose(
            scp_pearsonr(x2n[:, 1], x2n[:, 0]).statistic, pearsonr(x2n, axis=0)
        )
        self.assertAllClose(
            scp_pearsonr(y2n[:, 1], y2n[:, 0]).statistic, pearsonr(y2n, axis=0)
        )

    def test_pipe(self):
        rng = np.random.default_rng()
        x2n = rng.standard_normal((100, 2))
        y2n = rng.standard_normal((100, 2))
        f = PF.pearsonr(y=y2n[:, 0], axis=0)
        g = PF.pearsonr(y=x2n[:, 0], axis=0)
        self.assertAllClose(f(y2n[:, 1]), pearsonr(y2n, axis=0))
        self.assertAllClose(g(x2n[:, 1]), pearsonr(x2n, axis=0))


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
