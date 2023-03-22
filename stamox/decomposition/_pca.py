import functools as ft

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit


from ..core import Model


class PCA(Model):
    n_samples_: int
    n_features_: int
    n_components_: int
    componets_: jnp.array
    explained_variance_: jnp.array
    explained_variance_ratio_: jnp.array
    singular_values: jnp.array

    def __init__(self, n_components=None) -> None:
        super().__init__()
        self.n_components_ = n_components

    def fit(self, x, n_components=None):
        n_samples, n_features = x.shape
        self.n_samples_, self.n_features_ = n_samples, n_features
        if n_components is None:
            if self.n_components_ is None:
                ValueError("Need Number of Components")
        else:
            self.n_components = n_components
        U, S, Vt, self.componets_, self.explained_variance_, self.explained_variance_ratio_, self.singular_values = _pca(
            x, n_components)

        return self


@ft.partial(jit, static_argnums=1)
def _pca(x, n_components):
    n_samples, n_features = x.shape

    mean = jnp.mean(x, axis=0, keepdims=True)

    # Center data
    x_centered = x - mean

    U, S, Vt = jsp.linalg.svd(x_centered, full_matrices=False)

    components = Vt

    explained_variance = (S**2) / (n_samples - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var
    singular_values = S.copy()

    return U, S, Vt, components[:n_components], explained_variance[:n_components], explained_variance_ratio[:n_components], singular_values[:n_components]
