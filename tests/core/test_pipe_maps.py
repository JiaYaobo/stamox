"""Test Core Functions"""
import os

import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox import partial_pipe_pmap, partial_pipe_vmap, pipe_pmap, pipe_vmap


os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

class PipeMapTest(jtest.JaxTestCase):
    def test_vmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        h = pipe_vmap(lambda x: x + 1, in_axes=(0))
        y = X[:,0:] + 1
        self.assertAllClose(y, h(X))

    def test_pipe_vmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        h = pipe_vmap(lambda x: x + 1, in_axes=(0))
        g = pipe_vmap(lambda x: x + 2, in_axes=(0))
        f = g >> h
        y = X[:,0:] + 3
        self.assertAllClose(y, f(X))
    
    def test_partial_vmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        f = lambda x, y: x + y
        g = partial_pipe_vmap(f)
        z = X[:,0:] + 1
        self.assertAllClose(g(y=1.)(X), z)
    
    def test_partial_pipe_vmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        f = lambda x, y: x + y
        g = partial_pipe_vmap(f)
        h = partial_pipe_vmap(f)
        m = g(y=1.) >> h(y=2.)
        z = X[:,0:] + 3
        self.assertAllClose(m(X), z)
    
    def test_pmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        h = pipe_pmap(lambda x: x + 1, in_axes=(0))
        y = X[:,0:] + 1
        self.assertAllClose(y, h(X))
    
    def test_pipe_pmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        h = pipe_pmap(lambda x: x + 1, in_axes=(0))
        g = pipe_pmap(lambda x: x + 2, in_axes=(0))
        f = g >> h
        y = X[:,0:] + 3
        self.assertAllClose(y, f(X))
    
    def test_partial_pmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        f = lambda x, y: x + y
        g = partial_pipe_pmap(f)
        z = X[:,0:] + 1
        self.assertAllClose(g(y=1.)(X), z)
    
    def test_partial_pipe_pmap(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        f = lambda x, y: x + y
        h = partial_pipe_pmap(f)
        g = partial_pipe_pmap(f)
        m = g(y=1.) >> h(y=2.)
        z = X[:,0:] + 3
        self.assertAllClose(m(X), z)
    
    
if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
