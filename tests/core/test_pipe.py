"""Test For Pipe Functions"""
import jax.numpy as jnp
from absl.testing import absltest
from jax import grad, jacfwd, jacrev, jit, vmap
from jax._src import test_util as jtest

from stamox import make_partial_pipe, make_pipe, Pipe


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

    def test_pipe_jit(self):
        @make_pipe
        @jit
        def f(x):
            return x + 1

        @make_pipe
        @jit
        def g(x):
            return x * 2

        pipe = f >> g
        self.assertEqual(pipe(1), 4)

    def test_pipe_grad(self):
        @make_pipe
        @jit
        def f(x):
            return x + 1

        @make_pipe
        @jit
        def g(x):
            return x * 2

        pipe = f >> g
        self.assertEqual(grad(pipe)(1.0), 2.0)

    def test_pipe_jacfwd(self):
        @make_pipe
        @jit
        def f(x):
            return x + 1

        @make_pipe
        @jit
        def g(x):
            return x * 2

        pipe = f >> g
        self.assertEqual(jacfwd(pipe)(1.0), 2.0)

    def test_pipe_jacrev(self):
        @make_pipe
        @jit
        def f(x):
            return x + 1

        @make_pipe
        @jit
        def g(x):
            return x * 2

        pipe = f >> g
        self.assertEqual(jacrev(pipe)(1.0), 2.0)

    def test_pipe_vmap(self):
        @make_pipe
        @jit
        def f(x):
            return x + 1

        @make_pipe
        @jit
        def g(x):
            return x * 2

        pipe = f >> g
        self.assertEqual(vmap(pipe)(jnp.array([1.0])), 4.0)

    def test_partial_pipe_jit(self):
        @make_partial_pipe
        @jit
        def f(x, y):
            return x + y

        @make_pipe
        @jit
        def g(x):
            return x * 2

        pipe = f(y=2) >> g
        self.assertEqual(pipe(1), 6)

    def test_partial_pipe_grad(self):
        @make_partial_pipe
        @jit
        def f(x, y):
            return x + y

        @make_pipe
        @jit
        def g(x):
            return x * 2

        pipe = f(y=2) >> g
        self.assertEqual(grad(pipe)(1.0), 2.0)

    def test_partial_pipe_jacfwd(self):
        @make_partial_pipe
        @jit
        def f(x, y):
            return x + y

        @make_pipe
        @jit
        def g(x):
            return x * 2

        pipe = f(y=2) >> g
        self.assertEqual(jacfwd(pipe)(1.0), 2.0)

    def test_partial_pipe_jacrev(self):
        @make_partial_pipe
        @jit
        def f(x, y):
            return x + y

        @make_pipe
        @jit
        def g(x):
            return x * 2

        pipe = f(y=2) >> g
        self.assertEqual(jacrev(pipe)(1.0), 2.0)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
