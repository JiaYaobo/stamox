"""Tests for PCA."""
import numpy as np
import sklearn.decomposition
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.core import Pipeable
from stamox.decomposition import PCA


class PCATest(jtest.JaxTestCase):
    def test_pca(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        pca = PCA(n_components=2)(X)
        sk_pca = sklearn.decomposition.PCA(n_components=2).fit(X)
        self.assertArraysAllClose(np.abs(pca.components),np.abs(sk_pca.components_))
    
    def test_pipe_pca(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        pca = PCA(n_components=2)
        h = Pipeable(X) >> pca
        ans = h()
        sk_pca = sklearn.decomposition.PCA(n_components=2).fit(X)
        self.assertArraysAllClose(np.abs(ans.components),np.abs(sk_pca.components_))


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
