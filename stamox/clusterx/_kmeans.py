import functools as ft
import math

import jax
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jrand
import tensorflow_probability.substrates.jax.distributions as jd

from ..core import Model


class KMeans(Model):
    def __init__(self):
        super().__init__()

    def __call__(self, x, n_clusters, key=None):
        return kmeans_plusplus(x=x,
                               n_clusters=n_clusters,
                               key=key)


@ft.partial(jit, static_argnames=('n_clusters',))
def _kmeans_plusplus(x, n_clusters, *, key=None):
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
            seed=k0, sample_shape=(int(math.log(n_clusters) + 2), ))

        # chop to n - 1 samples
        candidate_ids = jnp.clip(
            candidate_ids, a_max=n_samples - 1)

        # Compute distances to center candidates
        distance_to_candidates = vmap(lambda x, y: euclidean_distance_square(x, y),
                                      in_axes=(0, None))(x[candidate_ids], x)
        # update closest distances to candidates
        distance_to_candidates = vmap(jnp.minimum, in_axes=(0, None))(
            distance_to_candidates, closest_dist_sq)

        candidates_pot = distance_to_candidates.sum(axis=-1)

        # Decide which candidate is the best
        best_candidate = jnp.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        carry = (current_pot, closest_dist_sq, k1)

        return carry, (x[best_candidate], best_candidate)

    init = (initial_pot, initial_closest_dist_sq, k1)
    _, (centers, indices) = lax.scan(
        _step, init, xs=None, length=n_clusters - 1)

    centers = jnp.vstack([initial_center, centers])
    indices = jnp.vstack([initial_indice, indices])

    return centers, indices


def kmeans_plusplus(x, n_clusters, key=None):
    if key is None:
        key = jrand.PRNGKey(0)

    return _kmeans_plusplus(x, n_clusters=n_clusters, key=key)


def _kmeans_single_lloyd(x, sample_wieght, centers_init, max_iter=300, tol=1e-4):

    n_clusters = centers_init.shape[0]

    centers = centers_init
    centers_new = jnp.zeros_like(centers)
    labels = jnp.full(x.shape[0], -1, dtype=jnp.int32)
    labels_old = labels.copy()
    weight_in_clusters = jnp.zeros(n_clusters, dtype=x.dtype)
    center_shift = jnp.zeros(n_clusters, dtype=x.dtype)

    

# def _kmeans_single_lloyd(
#     X,
#     sample_weight,
#     centers_init,
#     max_iter=300,
#     verbose=False,
#     x_squared_norms=None,
#     tol=1e-4,
#     n_threads=1,
# ):
#     """A single run of k-means lloyd, assumes preparation completed prior.
#     Parameters
#     ----------
#     X : {ndarray, sparse matrix} of shape (n_samples, n_features)
#         The observations to cluster. If sparse matrix, must be in CSR format.
#     sample_weight : ndarray of shape (n_samples,)
#         The weights for each observation in X.
#     centers_init : ndarray of shape (n_clusters, n_features)
#         The initial centers.
#     max_iter : int, default=300
#         Maximum number of iterations of the k-means algorithm to run.
#     verbose : bool, default=False
#         Verbosity mode
#     x_squared_norms : ndarray of shape (n_samples,), default=None
#         Precomputed x_squared_norms.
#     tol : float, default=1e-4
#         Relative tolerance with regards to Frobenius norm of the difference
#         in the cluster centers of two consecutive iterations to declare
#         convergence.
#         It's not advised to set `tol=0` since convergence might never be
#         declared due to rounding errors. Use a very small number instead.
#     n_threads : int, default=1
#         The number of OpenMP threads to use for the computation. Parallelism is
#         sample-wise on the main cython loop which assigns each sample to its
#         closest center.
#     Returns
#     -------
#     centroid : ndarray of shape (n_clusters, n_features)
#         Centroids found at the last iteration of k-means.
#     label : ndarray of shape (n_samples,)
#         label[i] is the code or index of the centroid the
#         i'th observation is closest to.
#     inertia : float
#         The final value of the inertia criterion (sum of squared distances to
#         the closest centroid for all observations in the training set).
#     n_iter : int
#         Number of iterations run.
#     """
#     n_clusters = centers_init.shape[0]

#     # Buffers to avoid new allocations at each iteration.
#     centers = centers_init
#     centers_new = np.zeros_like(centers)
#     labels = np.full(X.shape[0], -1, dtype=np.int32)
#     labels_old = labels.copy()
#     weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
#     center_shift = np.zeros(n_clusters, dtype=X.dtype)

#     if sp.issparse(X):
#         lloyd_iter = lloyd_iter_chunked_sparse
#         _inertia = _inertia_sparse
#     else:
#         lloyd_iter = lloyd_iter_chunked_dense
#         _inertia = _inertia_dense

#     strict_convergence = False

#     # Threadpoolctl context to limit the number of threads in second level of
#     # nested parallelism (i.e. BLAS) to avoid oversubscription.
#     with threadpool_limits(limits=1, user_api="blas"):
#         for i in range(max_iter):
#             lloyd_iter(
#                 X,
#                 sample_weight,
#                 x_squared_norms,
#                 centers,
#                 centers_new,
#                 weight_in_clusters,
#                 labels,
#                 center_shift,
#                 n_threads,
#             )

#             if verbose:
#                 inertia = _inertia(X, sample_weight, centers, labels, n_threads)
#                 print(f"Iteration {i}, inertia {inertia}.")

#             centers, centers_new = centers_new, centers

#             if np.array_equal(labels, labels_old):
#                 # First check the labels for strict convergence.
#                 if verbose:
#                     print(f"Converged at iteration {i}: strict convergence.")
#                 strict_convergence = True
#                 break
#             else:
#                 # No strict convergence, check for tol based convergence.
#                 center_shift_tot = (center_shift**2).sum()
#                 if center_shift_tot <= tol:
#                     if verbose:
#                         print(
#                             f"Converged at iteration {i}: center shift "
#                             f"{center_shift_tot} within tolerance {tol}."
#                         )
#                     break

#             labels_old[:] = labels

#         if not strict_convergence:
#             # rerun E-step so that predicted labels match cluster centers
#             lloyd_iter(
#                 X,
#                 sample_weight,
#                 x_squared_norms,
#                 centers,
#                 centers,
#                 weight_in_clusters,
#                 labels,
#                 center_shift,
#                 n_threads,
#                 update_centers=False,
#             )

#     inertia = _inertia(X, sample_weight, centers, labels, n_threads)

#     return labels, inertia, centers, i + 1
