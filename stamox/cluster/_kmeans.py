import jax.numpy as jnp
import jax.random as jrandom
from equinox import filter_jit
from jax import lax, vmap
from jax._src.random import KeyArray
from jax.scipy.cluster.vq import vq
from jaxtyping import ArrayLike, Int32

from ..core import Functional, StateFunc


class KMeansState(StateFunc):
    n_clusters: Int32
    centers: ArrayLike
    cluster: ArrayLike

    def __init__(self, n_clusters, centers, cluster):
        super().__init__(name="KMeans Cluster State", fn=None)
        self.n_clusters = n_clusters
        self.centers = centers
        self.cluster = cluster

    def _summary(self):
        return "KMeans Summary"


class KMeans(Functional):
    n_cluster: Int32
    key: KeyArray
    restarts: Int32

    def __init__(self, n_cluster=2, *, key, restarts=10):
        super().__init__(name="KMeans", fn=None)
        self.n_cluster = n_cluster
        self.key = key
        self.restarts = restarts

    def fit(self, X) -> KMeansState:
        return self._fit(X)

    def __call__(self, X, *args, **kwargs):
        return self._fit(X, *args, **kwargs)

    def _fit(self, X, *args, **kwargs):
        code_book, _ = kmeans_run(self.key, X, self.n_cluster, self.restarts)
        assignment, _ = vq(X, code_book)
        return KMeansState(
            n_clusters=self.n_cluster, centers=code_book, cluster=assignment
        )


@filter_jit
def kmeans_run(key, points, k, restarts):
    all_centroids, all_distortions = vmap(lambda key: _kmeans_run(key, points, k))(
        jrandom.split(key, restarts)
    )
    i = jnp.argmin(all_distortions)
    return all_centroids[i], all_distortions[i]


@filter_jit
def _kmeans_run(key, points, k, thresh=1e-5):
    def improve_centroids(val):
        prev_centroids, prev_distn, _ = val
        assignment, distortions = vq(points, prev_centroids)

        counts = (
            (assignment[jnp.newaxis, :] == jnp.arange(k)[:, jnp.newaxis])
            .sum(axis=1, keepdims=True)
            .clip(min=1)
        )
        new_centroids = jnp.sum(
            jnp.where(
                assignment[:, jnp.newaxis, jnp.newaxis]
                == jnp.arange(k)[jnp.newaxis, :, jnp.newaxis],
                points[:, jnp.newaxis, :],
                0.0,
            ).astype(dtype=jnp.float32),
            axis=0,
        ) / jnp.asarray(counts, dtype=jnp.float32)

        return new_centroids, jnp.mean(distortions), prev_distn

    initial_indices = jrandom.permutation(
        key, jnp.arange(points.shape[0]), independent=True
    )[:k]
    initial_val = improve_centroids((points[initial_indices, :], jnp.inf, None))
    centroids, distortion, _ = lax.while_loop(
        lambda val: (val[2] - val[1]) > thresh,
        improve_centroids,
        initial_val,
    )
    return centroids, distortion
