"""Test Core Functions"""
import os

import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox import pipe_pmap, pipe_vmap


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


class PipeMapTest(jtest.JaxTestCase):
    def test_vmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        h = pipe_vmap(lambda x: x + 1, in_axes=(0))
        y = X[:, 0:] + 1
        self.assertAllClose(y, h(X))

    def test_pipe_vmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        h = pipe_vmap(lambda x: x + 1, in_axes=(0))
        g = pipe_vmap(lambda x: x + 2, in_axes=(0))
        f = g >> h
        y = X[:, 0:] + 3
        self.assertAllClose(y, f(X))

    def test_pmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        h = pipe_pmap(lambda x: x + 1, in_axes=(0))
        y = X[:, 0:] + 1
        self.assertAllClose(y, h(X))

    def test_pipe_pmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        h = pipe_pmap(lambda x: x + 1, in_axes=(0))
        g = pipe_pmap(lambda x: x + 2, in_axes=(0))
        f = g >> h
        y = X[:, 0:] + 3
        self.assertAllClose(y, f(X))


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
