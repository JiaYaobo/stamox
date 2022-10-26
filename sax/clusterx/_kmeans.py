import functools as ft
import math

import jax
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jrand
import tensorflow_probability.substrates.jax.distributions as jd

from ..base import Model


class KMeans(Model):
    def __init__(self):
        super().__init__()

    def __call__(self, x, n_clusters, x_squared_norms=None, key=None):
        return kmeans_plusplus(x=x,
                               n_clusters=n_clusters,
                               x_squared_norms=x_squared_norms,
                               key=key)


@ft.partial(jit, static_argnums=[1, 2])
def _kmeans_plusplus(x, n_clusters, *, key=None):
    """Computational component for initialization of num_clusters by
    k-means++. Prior validation of data is assumed.
    https://github.com/scikit-learn/scikit-learn/blob/138619ae0b421826cf838af24627150fa8684cf5/sklearn/cluster/_kmeans.py#L161
    https://github.com/probml/dynamax/blob/main/dynamax/kmeans.py
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
    """
    n_samples, n_features = x.shape

    n_clusters = min(n_samples, n_clusters)

    k0, k1 = jrand.split(key)

    def euclidean_distance_square(x, y):
        return jnp.square(x - y).sum(axis=-1)

    center_id = jrand.randint(k0, (1,), minval=0, maxval=n_samples)[0]

    initial_center = x[center_id]

    initial_indice = center_id

    initial_closest_dist_sq = euclidean_distance_square(x[center_id], x)

    current_pot = initial_closest_dist_sq.sum()

    def _step(carry, inp=None):
        (_current_pot, _closest_dist_sq, key) = carry

        k0, k1 = jrand.split(key)

        candidate_ids = jd.Categorical(logits=jnp.log(_closest_dist_sq)).sample(
            seed=k0, sample_shape=(int(math.log(n_clusters) + 2), ))

        candidate_ids = jnp.clip(
            candidate_ids, a_max=n_samples - 1)

        # Compute distances to center candidates
        distance_to_candidates = vmap(lambda x, y: euclidean_distance_square(x, y),
                                      in_axes=(0, None))(x[candidate_ids], x)

        distance_to_candidates = vmap(jnp.minimum, in_axes=(0, None))(
            distance_to_candidates, _closest_dist_sq)

        candidates_pot = distance_to_candidates.sum(axis=-1)

        # Decide which candidate is the best
        best_candidate = jnp.argmin(candidates_pot)
        _current_pot = candidates_pot[best_candidate]
        _closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        carry = (_current_pot, _closest_dist_sq, k1)

        return carry, (x[best_candidate], best_candidate)

    init = (current_pot, initial_closest_dist_sq, k1)
    _, (centers, indices) = lax.scan(
        _step, init, xs=None, length=n_clusters - 1)

    centers = jnp.vstack([initial_center, centers])
    indices = jnp.vstack([initial_indice, indices])

    return centers, indices


def kmeans_plusplus(x, n_clusters, key=None):
    if key is None:
        key = jrand.PRNGKey(0)

    return _kmeans_plusplus(x, n_clusters=n_clusters, key=key)
