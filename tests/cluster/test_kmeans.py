"""Test for Kmeans"""
import jax.numpy as jnp
import jax.random
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.cluster import kmeans
from stamox.core import Pipeable, predict


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

        kms = kmeans(points, 3, key=k4)
        self.assertEqual(kms.centers.shape, (3, 2))

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

        kms = kmeans(n_cluster=3, key=k4)
        state = (Pipeable(points) >> kms)()
        self.assertEqual(state.centers.shape, (3, 2))
    
    def test_predict(self):
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

        kms = kmeans(n_cluster=3, key=k4)
        state = (Pipeable(points) >> kms)()
        self.assertEqual(state.centers.shape, (3, 2))
        self.assertEqual(predict(state, points).shape, (800,))


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
