import jax.numpy as jnp
from equinox import filter_jit
from jax import vmap
from jaxtyping import ArrayLike

from ...core import Functional, StateFunc


class PCAState(StateFunc):
    n_components: int
    components: ArrayLike
    mean: ArrayLike

    def __init__(self, n_components, components, mean):
        super().__init__(name="PCAState", fn=None)
        self.n_components = n_components
        self.components = components
        self.mean = mean
    


class PCA(Functional):
    n_components: int

    def __init__(self, n_components):
        super().__init__(name="PCA", fn=None)
        self.n_components = n_components

    def __call__(self, X, *args, **kwargs) -> PCAState:
        return self._pca(X)

    @filter_jit
    def _pca(self, X):
        # Subtract the mean of each feature column to center the data around the origin
        _mean = jnp.mean(X, axis=0)
        X_meaned = vmap(lambda x: x - _mean)(X)
        # Computing the covariance matrix
        cov_mat = jnp.cov(X_meaned.T)
        
        # Computing the eigenvalues and eigenvectors of the covariance matrix
        eigen_vals, eigen_vecs = jnp.linalg.eigh(cov_mat)
        
        # Sorting the eigenvalues in descending order and selecting the top n_components eigenvectors
        sorted_idx = eigen_vals.argsort()[::-1]
        sorted_eigen_vals = eigen_vals[sorted_idx]
        sorted_eigen_vecs = eigen_vecs[:, sorted_idx]
        eigenvectors_subset = sorted_eigen_vecs[:, :self.n_components]

        return PCAState(
            n_components=self.n_components, components=eigenvectors_subset, mean=_mean
        )
