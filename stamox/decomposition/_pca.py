import jax.numpy as jnp
from equinox import filter_jit
from jax import lax, vmap
from jaxtyping import ArrayLike

from ..core import StateFunc


class PCAState(StateFunc):
    n_components: int
    components: ArrayLike
    eigen_vals: ArrayLike
    mean: ArrayLike

    def __init__(self, n_components, components, eigen_vals, mean):
        super().__init__(name="PCAState", fn=None)
        self.n_components = n_components
        self.components = components
        self.mean = mean
        self.eigen_vals = eigen_vals


def princomp(x: ArrayLike, n_components: int) -> PCAState:
    """Performs principal component analysis (PCA) on the given array.

    Args:
        x (ArrayLike): The input array of shape (n_samples, n_features).
        n_components (int): Number of components to keep.

    Returns:
        PCAState: A namedtuple containing the results of the PCA.

    Example:
        >>> import jax.numpy as jnp
        >>> from stamox.functions import princomp
        >>> X = jnp.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=jnp.float32)
        >>> pca_state = princomp(X, n_components=2)
        >>> pca_state.components
        Array([[-0.8384922,  0.5449136],
            [-0.5449136, -0.8384922]], dtype=float32)
    """
    dtype = lax.dtype(x)
    x = jnp.asarray(x, dtype=dtype)
    n_components = int(n_components)
    if n_components < 1:
        raise ValueError(
            f"n_components={n_components} must be greater than or equal to 1"
        )
    if n_components > x.shape[1]:
        raise ValueError(
            f"n_components={n_components} must be less than or equal to the number of features={x.shape[1]}"
        )
    return _pca(x, n_components)


@filter_jit
def _pca(X, n_components):
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
    eigenvectors_subset = sorted_eigen_vecs[:, :n_components]

    return PCAState(
        n_components=n_components,
        components=eigenvectors_subset,
        mean=_mean,
        eigen_vals=sorted_eigen_vals,
    )
