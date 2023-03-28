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


# def kmeans_plusplus(x, n_clusters, key=None):
#     if key is None:
#         key = jrand.PRNGKey(0)

#     return _kmeans_plusplus(x, n_clusters=n_clusters, key=key)


# @ft.partial(jit, static_argnames=("n_clusters",))
# def _kmeans_plusplus(x: ArrayLike, n_clusters: Int32, *, key=None):
#     """Computational component for initialization of num_clusters by
#     k-means++. Prior validation of data is assumed.
#     Parameters
#     ----------
#     x : (num_samples, num_features)
#         The data to pick seeds for.
#     num_clusters : int
#         The number of seeds to choose.
#     Returns
#     -------
#     centers : ndarray of shape (num_clusters, num_features)
#         The initial centers for k-means.
#     indices : ndarray of shape (num_clusters, 1)
#         The initial centers for k-means.
#     References
#     ----------
#     https://github.com/scikit-learn/scikit-learn/blob/138619ae0b421826cf838af24627150fa8684cf5/sklearn/cluster/_kmeans.py#L161
#     https://github.com/probml/dynamax/blob/main/dynamax/kmeans.py
#     """
#     n_samples, n_features = x.shape

#     n_clusters = min(n_samples, n_clusters)

#     k0, k1 = jrand.split(key)

#     # def euclidean_distance_square(x, y):
#     #     return jnp.square(x - y).sum(axis=-1)

#     # # randomly choose a center
#     # center_id = jrand.randint(k0, (1,), minval=0, maxval=n_samples)[0]

#     # initial_center = x[center_id]

#     # initial_indice = center_id

#     # initial_closest_dist_sq = euclidean_distance_square(x[center_id], x)

#     # initial_pot = initial_closest_dist_sq.sum()

#     # def _step(carry, inp=None):
#     #     (current_pot, closest_dist_sq, key) = carry

#     #     k0, k1 = jrand.split(key)

#     #     # use distances as probabilities
#     #     candidate_ids = jd.Categorical(logits=jnp.log(closest_dist_sq)).sample(
#     #         seed=k0, sample_shape=(int(math.log(n_clusters) + 2),)
#     #     )

#     #     # chop to n - 1 samples
#     #     candidate_ids = jnp.clip(candidate_ids, a_max=n_samples - 1)

#     #     # Compute distances to center candidates
#     #     distance_to_candidates = vmap(
#     #         lambda x, y: euclidean_distance_square(x, y), in_axes=(0, None)
#     #     )(x[candidate_ids], x)
#     #     # update closest distances to candidates
#     #     distance_to_candidates = vmap(jnp.minimum, in_axes=(0, None))(
#     #         distance_to_candidates, closest_dist_sq
#     #     )

#     #     candidates_pot = distance_to_candidates.sum(axis=-1)

#     #     # Decide which candidate is the best
#     #     best_candidate = jnp.argmin(candidates_pot)
#     #     current_pot = candidates_pot[best_candidate]
#     #     closest_dist_sq = distance_to_candidates[best_candidate]
#     #     best_candidate = candidate_ids[best_candidate]

#     #     carry = (current_pot, closest_dist_sq, k1)

#     #     return carry, (x[best_candidate], best_candidate)

#     # init = (initial_pot, initial_closest_dist_sq, k1)
#     # _, (centers, indices) = lax.scan(_step, init, xs=None, length=n_clusters - 1)

#     # centers = jnp.vstack([initial_center, centers])
#     # indices = jnp.vstack([initial_indice, indices])

#     # return centers, indices
