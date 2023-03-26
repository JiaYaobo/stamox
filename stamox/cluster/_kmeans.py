import functools as ft
import math

from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jrand
from jaxtyping import Int32, PyTree, ArrayLike
import tensorflow_probability.substrates.jax.distributions as jd

from ..core import StateFunc


class KMeans(StateFunc):
    n_clusters: Int32
    max_iters: Int32
    _params: PyTree

    def __init__(
        self, n_clusters: Int32 = 2, max_iters: Int32 = 10, method: str = "kmeans++"
    ):
        super().__init__(name="KMeans Cluster")
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self._params = {
            "cluster": jnp.array([], dtype=jnp.float32),
            "centers": jnp.array([], dtype=jnp.float32),
            "totss": 0.,
            "withinss": 0.,
            "betweenss": 0.,
        }

    def _kmeans_plusplus(self, x):
        pass

    def __call__(self, x, *, key=None):
        return kmeans_plusplus(x=x, n_clusters=self.n_clusters, key=key)


@ft.partial(jit, static_argnames=("n_clusters",))
def _kmeans_plusplus(x: ArrayLike, n_clusters: Int32, *, key=None):
    """Computational component for initialization of num_clusters by
    k-means++. Prior validation of data is assumed.
    Parameters
    ----------
    x : (num_samples, num_features)
        The data to pick seeds for.
    num_clusters : int
        The number of seeds to choose.
    Returns
    -------
    centers : ndarray of shape (num_clusters, num_features)
        The initial centers for k-means.
    indices : ndarray of shape (num_clusters, 1)
        The initial centers for k-means.
    References
    ----------
    https://github.com/scikit-learn/scikit-learn/blob/138619ae0b421826cf838af24627150fa8684cf5/sklearn/cluster/_kmeans.py#L161
    https://github.com/probml/dynamax/blob/main/dynamax/kmeans.py
    """
    n_samples, n_features = x.shape

    n_clusters = min(n_samples, n_clusters)

    k0, k1 = jrand.split(key)

    def euclidean_distance_square(x, y):
        return jnp.square(x - y).sum(axis=-1)

    # randomly choose a center
    center_id = jrand.randint(k0, (1,), minval=0, maxval=n_samples)[0]

    initial_center = x[center_id]

    initial_indice = center_id

    initial_closest_dist_sq = euclidean_distance_square(x[center_id], x)

    initial_pot = initial_closest_dist_sq.sum()

    def _step(carry, inp=None):
        (current_pot, closest_dist_sq, key) = carry

        k0, k1 = jrand.split(key)

        # use distances as probabilities
        candidate_ids = jd.Categorical(logits=jnp.log(closest_dist_sq)).sample(
            seed=k0, sample_shape=(int(math.log(n_clusters) + 2),)
        )

        # chop to n - 1 samples
        candidate_ids = jnp.clip(candidate_ids, a_max=n_samples - 1)

        # Compute distances to center candidates
        distance_to_candidates = vmap(
            lambda x, y: euclidean_distance_square(x, y), in_axes=(0, None)
        )(x[candidate_ids], x)
        # update closest distances to candidates
        distance_to_candidates = vmap(jnp.minimum, in_axes=(0, None))(
            distance_to_candidates, closest_dist_sq
        )

        candidates_pot = distance_to_candidates.sum(axis=-1)

        # Decide which candidate is the best
        best_candidate = jnp.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        carry = (current_pot, closest_dist_sq, k1)

        return carry, (x[best_candidate], best_candidate)

    init = (initial_pot, initial_closest_dist_sq, k1)
    _, (centers, indices) = lax.scan(_step, init, xs=None, length=n_clusters - 1)

    centers = jnp.vstack([initial_center, centers])
    indices = jnp.vstack([initial_indice, indices])

    return centers, indices


def kmeans_plusplus(x, n_clusters, key=None):
    if key is None:
        key = jrand.PRNGKey(0)

    return _kmeans_plusplus(x, n_clusters=n_clusters, key=key)
