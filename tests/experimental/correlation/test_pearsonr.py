"""Test Pearson correlation coefficient"""
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import pearsonr as scp_pearsonr

from stamox.experimental import pearsonr


class PearsonrTest(jtest.JaxTestCase):
    def test_pearsonr_two_side(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y = np.array([5, 6, 7, 8, 7], dtype=np.float32)
        r = pearsonr(x, y)
        scp_r = scp_pearsonr(x, y)
        self.assertAllClose(r[0], scp_r[0], atol=1e-6)
        self.assertAllClose(r[1][0], scp_r[1], atol=1e-6)

    def test_pearsonr_less(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y = np.array([5, 6, 7, 8, 7], dtype=np.float32)
        r = pearsonr(x, y, alternative="less")
        scp_r = scp_pearsonr(x, y, alternative="less")
        self.assertAllClose(r[0], scp_r[0], atol=1e-6)
        self.assertAllClose(r[1][0], scp_r[1], atol=1e-6)
    
    def test_pearsonr_greater(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y = np.array([5, 6, 7, 8, 7], dtype=np.float32)
        r = pearsonr(x, y, alternative="greater")
        scp_r = scp_pearsonr(x, y, alternative="greater")
        self.assertAllClose(r[0], scp_r[0], atol=1e-6)
        self.assertAllClose(r[1][0], scp_r[1], atol=1e-6)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
