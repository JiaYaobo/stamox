"""Test Base Functions"""
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox import Functional, StateFunc


class BaseTest(jtest.JaxTestCase):
    def test_functional(self):

        def f(x):
            return x + 1

        func = Functional(f)
        self.assertEqual(func(1), 2)

    def test_state_func(self):
        def f(x):
            return x + 1

        func = StateFunc(f)
        self.assertEqual(func(1), 2)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())