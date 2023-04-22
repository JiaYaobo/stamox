"""Test Correlation coefficient"""
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import pearsonr as scp_pearsonr, spearmanr as scp_spearman

import stamox.pipe_functions as PF
from stamox.correlation import cor


class CorTest(jtest.JaxTestCase):
    def test_pearson_correlation_1d(self):
        x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9], np.float32)
        y = jnp.array([1, 4, 3, 6, 5, 6, 8, 8, 9], np.float32)
        self.assertAllClose(scp_pearsonr(x, x).statistic.astype(np.float32), cor(x, x))
        self.assertAllClose(scp_pearsonr(x, y).statistic.astype(np.float32), cor(x, y))
        self.assertAllClose(
            scp_pearsonr(x, y).statistic.astype(np.float32), cor([x, y], axis=1)
        )

    def test_pearson_correlation_2d(self):
        rng = np.random.default_rng()
        x2n = rng.standard_normal((100, 2))
        y2n = rng.standard_normal((100, 2))
        self.assertAllClose(
            scp_pearsonr(x2n[:, 1], x2n[:, 0]).statistic, cor(x2n, axis=0)
        )
        self.assertAllClose(
            scp_pearsonr(y2n[:, 1], y2n[:, 0]).statistic, cor(y2n, axis=0)
        )

    def test_spearman_correlation_1d(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], np.float32)
        y = np.array([1, 4, 3, 6, 5, 6, 8, 8, 9], np.float32)
        self.assertAllClose(scp_spearman(x, x).statistic, cor(x, x, method="spearman"))
        self.assertAllClose(scp_spearman(x, y).statistic, cor(x, y, method="spearman"))
        self.assertAllClose(
            scp_spearman(x, y).statistic, cor([x, y], axis=1, method="spearman")
        )

    def test_spearman_correlation_2d(self):
        rng = np.random.default_rng()
        x2n = rng.standard_normal((100, 2))
        y2n = rng.standard_normal((100, 2))
        self.assertAllClose(
            scp_spearman(x2n[:, 1], x2n[:, 0]).statistic,
            cor(x2n, axis=0, method="spearman"),
        )
        self.assertAllClose(
            scp_spearman(y2n[:, 1], y2n[:, 0]).statistic,
            cor(y2n, axis=0, method="spearman"),
        )

    def test_pipe(self):
        rng = np.random.default_rng()
        x2n = rng.standard_normal((100, 2))
        y2n = rng.standard_normal((100, 2))
        f = PF.cor(y=y2n[:, 0], axis=0)
        g = PF.cor(y=x2n[:, 0], axis=0)
        f1 = PF.cor(y=y2n[:, 0], axis=0, method="spearman")
        g1 = PF.cor(y=x2n[:, 0], axis=0, method="spearman")
        self.assertAllClose(f(y2n[:, 1]), cor(y2n, axis=0))
        self.assertAllClose(g(x2n[:, 1]), cor(x2n, axis=0))
        self.assertAllClose(f1(y2n[:, 1]), cor(y2n, axis=0, method="spearman"))
        self.assertAllClose(g1(x2n[:, 1]), cor(x2n, axis=0, method="spearman"))


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
