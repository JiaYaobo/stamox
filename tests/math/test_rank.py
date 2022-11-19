"""Test for beta distribution"""

import jax.random as jrand
import jax.numpy as jnp

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtest

from stamox.math.rank import rank, rank_fast_on_cpu, rank_fast_on_gpu



class RankTest(jtest.JaxTestCase):
    def test_rank_without_knots(self):
       x = np.array([2., 3., 1., 4., 9. ,7.])
       ranks = rank_fast_on_gpu(x)
       true_ranks = rank_fast_on_cpu(x)
       self.assertAllClose(ranks, true_ranks)

    
    def test_rank_with_knots(self):
       x = np.array([2, 2, 3, 1, 4, 9, 7, 7], dtype=np.float32)
       ranks = rank_fast_on_gpu(x)
       true_ranks = rank_fast_on_cpu(x)
       self.assertAllClose(ranks, true_ranks)

if __name__ == '__main__':
    absltest.main(testLoader=jtest.JaxTestLoader())