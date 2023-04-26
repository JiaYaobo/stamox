"""Tests for princomp."""
import numpy as np
import sklearn.decomposition

from stamox import Pipeable
from stamox.pipe_functions import princomp


def test_pca():
    X = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
    )
    pca = princomp(n_components=2)(X)
    sk_pca = sklearn.decomposition.PCA(n_components=2).fit(X)
    np.testing.assert_allclose(np.abs(pca.components), np.abs(sk_pca.components_))


def test_pipe_pca():
    X = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
    )
    pca = princomp(n_components=2)
    h = Pipeable(X) >> pca
    ans = h()
    sk_pca = sklearn.decomposition.PCA(n_components=2).fit(X)
    np.testing.assert_allclose(np.abs(ans.components), np.abs(sk_pca.components_))
