"""Test for Kmeans"""
import jax.numpy as jnp
import jax.random
from absl.testing import absltest
from equinox import filter_jit
from jax._src import test_util as jtest

from stamox.cluster import KMeans
from stamox.core import Pipeable


class KMeansTest(jtest.JaxTestCase):
    def test_kmeans(self):
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(20010218), 4)

        points = jnp.concatenate(
            [
                jax.random.normal(k1, (400, 2))
                + jnp.array([[4, 0]], dtype=jnp.float32),
                jax.random.normal(k2, (200, 2))
                + jnp.array([[0.5, 1]], dtype=jnp.float32),
                jax.random.normal(k3, (200, 2))
                + jnp.array([[-0.5, -1]], dtype=jnp.float32),
            ]
        )

        kms = KMeans(3, key=k4)
        state = kms(points)
        self.assertEqual(state.centers.shape, (3, 2))

    def test_pipe_kmeans(self):
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(20010218), 4)

        points = jnp.concatenate(
            [
                jax.random.normal(k1, (400, 2))
                + jnp.array([[4, 0]], dtype=jnp.float32),
                jax.random.normal(k2, (200, 2))
                + jnp.array([[0.5, 1]], dtype=jnp.float32),
                jax.random.normal(k3, (200, 2))
                + jnp.array([[-0.5, -1]], dtype=jnp.float32),
            ]
        )

        kms = KMeans(3, key=k4)
        state = (Pipeable(points) >> kms)()
        self.assertEqual(state.centers.shape, (3, 2))

    def test_pipe_kmeans_jit(self):
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(20010218), 4)

        points = jnp.concatenate(
            [
                jax.random.normal(k1, (400, 2))
                + jnp.array([[4, 0]], dtype=jnp.float32),
                jax.random.normal(k2, (200, 2))
                + jnp.array([[0.5, 1]], dtype=jnp.float32),
                jax.random.normal(k3, (200, 2))
                + jnp.array([[-0.5, -1]], dtype=jnp.float32),
            ]
        )

        kms = KMeans(3, key=k4)
        state = filter_jit(Pipeable(points) >> kms)()
        self.assertEqual(state.centers.shape, (3, 2))


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
