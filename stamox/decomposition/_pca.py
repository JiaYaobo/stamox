import jax.numpy as np
from equinox import filter_jit

from ..core import StateFunc, Functional
from ..basic import mean


class PCAState(StateFunc):
    n_components: int
    components: np.array
    mean: np.array

    def __init__(self, n_components, components, mean):
        super().__init__(name='PCAState', fn=None)
        self.n_components = n_components
        self.components = components
        self.mean = mean

    def _summary(self) -> str:
        return "PCA Summary"


class PCA(Functional):
    n_components: int

    def __init__(self, n_components):
        super().__init__(name="PCA", fn=None)
        self.n_components = n_components

    def __call__(self, x, *args, **kwargs):
        return _pca(x, n_components=self.n_components)


@filter_jit
def _pca(X, n_components):
    # Subtract the mean of each feature column to center the data around the origin
    _mean = mean(X, axis=0)
    X = X - _mean

    # Compute the covariance matrix of the centered input data
    cov_matrix = np.cov(X.T)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvectors in decreasing order of their corresponding eigenvalues
    idxs = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idxs]

    # Store only the top k eigenvectors that capture most of the variation in the data
    _components = eigenvectors[:, : n_components]

    return PCAState(n_components=n_components, components=_components, mean=_mean)


