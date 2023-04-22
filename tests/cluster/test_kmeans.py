"""Test for Kmeans"""
import jax.numpy as jnp
import jax.random
import numpy as np

import stamox.pipe_functions as PF
from stamox.cluster import kmeans
from stamox.core import Pipeable, predict


def test_kmeans():
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(20010218), 4)

    points = jnp.concatenate(
        [
            jax.random.normal(k1, (400, 2)) + jnp.array([[4, 0]]),
            jax.random.normal(k2, (200, 2)) + jnp.array([[0.5, 1]]),
            jax.random.normal(k3, (200, 2)) + jnp.array([[-0.5, -1]]),
        ]
    )

    kms = kmeans(points, 3, key=k4)
    np.testing.assert_equal(kms.centers.shape, (3, 2))


def test_pipe_kmeans():
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(20010218), 4)

    points = jnp.concatenate(
        [
            jax.random.normal(k1, (400, 2)) + jnp.array([[4, 0]]),
            jax.random.normal(k2, (200, 2)) + jnp.array([[0.5, 1]]),
            jax.random.normal(k3, (200, 2)) + jnp.array([[-0.5, -1]]),
        ]
    )

    kms = PF.kmeans(n_cluster=3, key=k4)
    state = (Pipeable(points) >> kms)()
    np.testing.assert_equal(state.centers.shape, (3, 2))


def test_predict():
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(20010218), 4)

    points = jnp.concatenate(
        [
            jax.random.normal(k1, (400, 2)) + jnp.array([[4, 0]]),
            jax.random.normal(k2, (200, 2)) + jnp.array([[0.5, 1]]),
            jax.random.normal(k3, (200, 2)) + jnp.array([[-0.5, -1]]),
        ]
    )

    kms = PF.kmeans(n_cluster=3, key=k4)
    state = (Pipeable(points) >> kms)()
    np.testing.assert_equal(state.centers.shape, (3, 2))
    np.testing.assert_equal(predict(points, state).shape, (800,))
