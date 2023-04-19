"""Test better partial"""
import jax.numpy as jnp
from absl.testing import absltest
from jax import jit, vmap
from jax._src import test_util as jtest

from stamox.experimental import better_partial


class BetterPartialTest(jtest.JaxTestCase):
    def test_better_partial1(self):
        def f(a, b, c, d):
            return a + b + c + d

        g = better_partial(f, 1, 2, 3)
        self.assertEqual(g(4), 10)
        self.assertEqual(g(5), 11)

    def test_better_partial2(self):
        def f(a, b, c, d):
            return a + b + c + d

        g = better_partial(f, b=1, a=2, c=3)
        self.assertEqual(g(4), 10)
        self.assertEqual(g(5), 11)

    def test_better_partial3(self):
        def f(a, b, c, d):
            return a * b + c * d

        g = better_partial(f, b=2, c=3)
        self.assertEqual(g(1, 2), 8)

    def test_better_partial4(self):
        def f(a, b=None, c=None, d=None):
            return a * b + c * d

        g = better_partial(f, b=2, c=3)
        self.assertEqual(g(1, 2), 8)

    def test_better_partial5(self):
        def f(a, b, c, d):
            return a * b + c * d

        g = better_partial(f, c=1)
        self.assertEqual(g(1, 2, 3), 5)

    def test_jit_compatible(self):
        def f(a, b, c, d):
            return a * b + c * d

        f = jit(f)
        g = better_partial(f, 1, 2, 3)
        jit_g = jit(g)
        self.assertEqual(g(4), 14)
        self.assertEqual(jit_g(4), 14)

    def test_jit_compatible2(self):
        def f(a, b, c, d):
            return a * b + c * d

        f = jit(f)
        g = better_partial(f, b=1)
        jit_g = jit(g)
        self.assertEqual(g(2, 3, 4), 14)
        self.assertEqual(jit_g(2, 3, 4), 14)

    def test_vmap_compatible(self):
        def f(a, b):
            return a * b

        g = better_partial(f, 1)
        vmap_g = vmap(g)
        b = jnp.arange(10)
        self.assertArraysEqual(vmap_g(b), b)

    def test_vmap_compatible2(self):
        def f(a, b, c):
            return a * b + c

        g = better_partial(f, b=1)
        vmap_g = vmap(g)
        a = jnp.arange(10)
        c = jnp.arange(10)
        self.assertArraysEqual(vmap_g(a, c), a + c)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
