import functools as ft
import math

import jax
from jax import jit, lax
import jax.numpy as jnp
import jax.random as jrand

from ..base import Model
from ..math._euclid import _euclidean_distances



class KMeans(Model):
    def __init__(self):
        super().__init__()

    def __call__(self, x, n_clusters, x_squared_norms=None, key=None):
        return kmeans_plusplus(x=x,
                               n_clusters=n_clusters,
                               x_squared_norms=x_squared_norms,
                               key=key)


@ft.partial(jax.jit, static_argnums=[1, 2])
def _kmeans_plusplus(x, n_clusters, x_squared_norms=None, *, key=None):
    x = jnp.asarray(x)

    n_samples, n_features = x.shape

    centers = jnp.zeros((n_clusters, n_features))

    center_id = jrand.randint(key, (1,), minval=0, maxval=n_samples)[0]
    indices = jnp.full((n_clusters,), -1, dtype=jnp.int32)

    centers = centers.at[0].set(x[center_id])
    indices = indices.at[0].set(center_id)

    closest_dist_sq = _euclidean_distances(centers[0, jnp.newaxis],
                                           x,
                                           y_norm_squared=x_squared_norms,
                                           squared=True)

    current_pot = closest_dist_sq.sum()

    def _step(carry, inp=None):
        (i, _centers, _indices, _current_pot, _closest_dist_sq) = carry

        rand_vals = jrand.uniform(key=key, shape=(int(math.log(n_clusters)) + 2,)) * _current_pot

        candidate_ids = jnp.searchsorted(jnp.cumsum(_closest_dist_sq), rand_vals)

        candidate_ids = jnp.clip(candidate_ids, None, _closest_dist_sq.size - 1)

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(
            x[candidate_ids], x, y_norm_squared=x_squared_norms, squared=True
        )

        distance_to_candidates = jnp.minimum(_closest_dist_sq, distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = jnp.argmin(candidates_pot)
        _current_pot = candidates_pot[best_candidate]
        _closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        _centers = _centers.at[i].set(x[best_candidate])
        _indices = _indices.at[i].set(best_candidate)

        carry = (i + 1, _centers, _indices, _current_pot, _closest_dist_sq)

        return carry, (_centers, _indices)

    init = (1, centers, indices, current_pot, closest_dist_sq[0])
    _, (centers, indices) = lax.scan(_step, init, xs=None, length=n_clusters - 1)

    return centers, indices


def kmeans_plusplus(x, n_clusters, x_squared_norms=None, key=None):
    if key is None:
        key = jrand.PRNGKey(0)

    return _kmeans_plusplus(x, n_clusters=n_clusters, x_squared_norms=x_squared_norms, key=key)
