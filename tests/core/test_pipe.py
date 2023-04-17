"""Test For Pipe Functions"""
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.core import make_partial_pipe, make_pipe, Pipe


class PipeTest(jtest.JaxTestCase):
    def test_create_pipe_class(self):
        @make_pipe
        def f(x):
            return x + 1

        def g(x):
            return x * 2

        pipe = f >> g
        self.assertIsInstance(pipe, Pipe)

    def test_iter_pipe_class(self):
        @make_pipe
        def f(x):
            return x + 1

        def g(x):
            return x * 2

        pipe = f >> g
        self.assertEqual(len(list(pipe)), 2)

    def test_getitem_pipe_class(self):
        @make_pipe
        def f(x):
            return x + 1

        def g(x):
            return x * 2

        pipe = f >> g
        self.assertEqual(pipe[0].func, f.func)
        self.assertEqual(pipe[1].func, g)

    def test_slice_pipe_class(self):
        @make_pipe
        def f(x):
            return x + 1

        def g(x):
            return x * 2

        pipe = f >> g >> f >> g >> f >> g
        self.assertEqual(len(pipe[0:1]), 1)
        self.assertEqual(pipe[0:1][0].func, f.func)

    def test_create_partial_pipe_class(self):
        @make_partial_pipe
        def f(x, y):
            return x + y

        def g(x):
            return x * 2

        pipe = f(y=2) >> g
        self.assertIsInstance(pipe, Pipe)

    def test_evaluate_partial_pipe_class(self):
        @make_partial_pipe
        def f(x, y):
            return x + y

        def g(x):
            return x * 2

        pipe1 = f(y=2) >> g
        self.assertEqual(pipe1(1), 6)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
