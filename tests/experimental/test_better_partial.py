"""Test better partial"""
from absl.testing import absltest
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
            return a * b + c * d

        g = better_partial(f, b=1, a=2, d=4)
        self.assertEqual(g(3), 14)
        self.assertEqual(g(5), 22)
    
    def test_better_partial3(self):
        def f(a, b, c, d):
            return a * b + c * d

        g = better_partial(f, 1, 2, b=4)
        self.assertEqual(g(3), 10)
        self.assertEqual(g(5), 14)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
