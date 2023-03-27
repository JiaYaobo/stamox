"""Test for Combination"""
from absl.testing import absltest

import jax.numpy as jnp
import numpy as np
from jax._src import test_util as jtest

from stamox.math.combination import choose


class CombinationTest(jtest.JaxTestCase):
    def test_comb(self):
        n = 5
        k = np.array([0, 1, 2, 3, 4, 5, 6, -1], dtype=np.int32)
        combs = choose(k, n)
        true_combs = np.array([1, 5, 10, 10, 5, 1, 0, 0], dtype=jnp.int32)
        self.assertArraysEqual(combs, true_combs)

    def test_pick(self):
        n = 5
        k = np.array([0, 1, 2, 3, 4, 5, 6, -1], dtype=np.int32)
        combs = choose(n=n)(k)
        true_combs = np.array([1, 5, 10, 10, 5, 1, 0, 0], dtype=jnp.int32)
        self.assertArraysEqual(combs, true_combs)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
