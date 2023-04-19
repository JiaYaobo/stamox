import jax.numpy as jnp
import jax.random as jrandom
from equinox import filter_jit
from jax import lax, vmap
from jax._src.random import KeyArray
from jax.scipy.cluster.vq import vq
from jaxtyping import ArrayLike

from ..core import StateFunc


class KMeansState(StateFunc):
    """KMeansState class for K-means clustering.

    Attributes:
        n_clusters (int): Number of clusters.
        centers (ArrayLike): Centers of the clusters.
        cluster (ArrayLike): Cluster labels for each point.
        iters (int): Number of iterations.
        totss (float): Total sum of squares.
        betwss (float): Between sum of squares.
        withinss (float): Within sum of squares.
        tot_withinss (float): Total within sum of squares.
    """

    n_clusters: int
    centers: ArrayLike
    cluster: ArrayLike
    iters: int
    totss: float
    betwss: float
    withinss: float
    tot_withinss: float

    def __init__(
        self, n_clusters, centers, cluster, iters, totss, betwss, withinss, tot_withinss
    ):
        super().__init__(name="KMeans Cluster State", fn=None)
        self.n_clusters = n_clusters
        self.centers = centers
        self.cluster = cluster
        self.iters = iters
        self.totss = totss
        self.betwss = betwss
        self.withinss = withinss
        self.tot_withinss = tot_withinss

    def _predict(self, x: ArrayLike):
        # predict the cluster for new data
        return vq(x, self.centers)[0]


def kmeans(
    x: ArrayLike,
    n_cluster: int,
    restarts: int = 10,
    max_iters: int = 100,
    dtype: jnp.dtype = jnp.float32,
    *,
    key: KeyArray = None
):
    """Runs the K-means clustering algorithm on a given dataset.

    Args:
        x (ArrayLike): The dataset to be clustered.
        n_cluster (int): The number of clusters to generate.
        restarts (int, optional): The number of restarts for the algorithm. Defaults to 10.
        max_iters (int, optional): The maximum number of iterations for the algorithm. Defaults to 100.
        dtype (jnp.dtype, optional): The data type of the output. Defaults to jnp.float32.
        key (KeyArray, optional): A key array used for encryption. Defaults to None.

    Returns:
        KMeansState: An object containing the results of the clustering algorithm.

    Example:
        >>> from jax import random
        >>> key = random.PRNGKey(0)
        >>> x = random.normal(key, shape=(100, 2))
        >>> state = kmeans(x, n_cluster=3, restarts=5, max_iters=50, key=key)
        >>> state.centers
        Array([[ 0.8450022 , -1.0791471 ],
                     [-0.7179966 ,  0.6372063 ],
                     [ 0.09818084, -0.25906876]], dtype=float32)
    """
    x = jnp.asarray(x, dtype=dtype)
    code_book, _, iters = kmeans_run(key, x, n_cluster, max_iters, restarts)
    assignment, _ = vq(x, code_book)
    totss = jnp.sum((x - jnp.mean(x, axis=0).reshape(1, -1)) ** 2)
    # Vector of within-cluster sum of squares, one component per cluster.
    withinss = jnp.array(  # pylint: disable=invalid-unary-operand-type
        [
            jnp.sum((x[assignment == i] - code_book[i].reshape(1, -1)) ** 2)
            for i in range(n_cluster)
        ]
    )
    # The between-cluster sum of squares,
    betwss = totss - jnp.sum(withinss)
    tot_withinss = jnp.sum(withinss)

    return KMeansState(
        n_clusters=n_cluster,
        centers=code_book,
        cluster=assignment,
        iters=iters,
        totss=totss,
        betwss=betwss,
        withinss=withinss,
        tot_withinss=tot_withinss,
    )


@filter_jit
def kmeans_run(key, points, k, max_iters, restarts):
    all_centroids, all_distortions, iters = vmap(
        lambda key: _kmeans_run(key, points, k, max_iters)
    )(jrandom.split(key, restarts))
    i = jnp.argmin(all_distortions)
    return all_centroids[i], all_distortions[i], iters[i]


@filter_jit
def _kmeans_run(key, points, k, max_iters, thresh=1e-5):
    def improve_centroids(val):
        prev_centroids, prev_distn, _, i = val
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

        return new_centroids, jnp.mean(distortions), prev_distn, i + 1

    initial_indices = jrandom.permutation(
        key, jnp.arange(points.shape[0]), independent=True
    )[:k]
    initial_val = improve_centroids((points[initial_indices, :], jnp.inf, None, 0))
    centroids, distortion, _, iters = lax.while_loop(
        lambda val: ((val[2] - val[1]) > thresh) & (val[3] < max_iters),
        improve_centroids,
        initial_val,
    )
    return centroids, distortion, iters
