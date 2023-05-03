"""Test Pipe Jit Functions"""
import equinox as eqx
import jax.numpy as jnp
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox import make_partial_pipe, pipe_jit, pipe_vmap


class PipeJitTest(jtest.JaxTestCase):
    def test_jit_jit(self):
        num_traces = 0

        @pipe_jit
        @pipe_jit
        def f(x):
            nonlocal num_traces
            num_traces += 1
            return x + 1

        assert f(jnp.array(1)) == 2
        assert f(jnp.array(2)) == 3
        assert num_traces == 1

        @pipe_jit
        def g(x):
            nonlocal num_traces
            num_traces += 1
            return x + 1

        assert g(jnp.array(1)) == 2
        assert g(jnp.array(2)) == 3
        assert num_traces == 2

    def test_jit_grad(self):
        num_traces = 0

        def f(x):
            nonlocal num_traces
            num_traces += 1
            return x + 1

        assert pipe_jit(eqx.filter_grad(f))(jnp.array(1.0)) == 1
        assert pipe_jit(eqx.filter_grad(f))(jnp.array(2.0)) == 1
        assert num_traces == 1

        assert pipe_jit(eqx.filter_value_and_grad(f))(jnp.array(1.0)) == (2, 1)
        assert pipe_jit(eqx.filter_value_and_grad(f))(jnp.array(2.0)) == (3, 1)
        assert num_traces == 2

    def test_jit_vmap(self):
        num_traces = 0

        def f(x):
            nonlocal num_traces
            num_traces += 1
            return x + 1

        out = pipe_jit(pipe_vmap(f))(jnp.array([1, 2]))
        self.assertAllClose(out, jnp.array([2, 3]))
        assert num_traces == 1

        out = pipe_jit(pipe_vmap(f))(jnp.array([2, 3]))
        self.assertAllClose(out, jnp.array([3, 4]))
        assert num_traces == 1

    def test_partial_jit_jit(self):
        num_traces = 0

        @make_partial_pipe
        @pipe_jit
        def f(x, y):
            nonlocal num_traces
            num_traces += 1
            return x + y

        assert f(x=jnp.array(1))(1) == 2
        assert f(x=jnp.array(2))(1) == 3
        assert num_traces == 1

        @make_partial_pipe
        @pipe_jit
        def g(x):
            nonlocal num_traces
            num_traces += 1
            return x + 1

        assert g(jnp.array(1)) == 2
        assert g(jnp.array(2)) == 3
        assert num_traces == 2

    def test_partial_jit_grad(self):
        num_traces = 0

        @make_partial_pipe
        @pipe_jit
        @eqx.filter_grad
        def f(x, y):
            nonlocal num_traces
            num_traces += 1
            return y * x + 1

        self.assertEqual(f(jnp.array(1.0), y=2.0), 2)
        self.assertEqual(f(jnp.array(2.0), y=2.0), 2)
        assert num_traces == 1

        def g(x, y):
            nonlocal num_traces
            num_traces += 1
            return y * x + 1

        self.assertEqual(
            pipe_jit(eqx.filter_value_and_grad(g))(jnp.array(1.0), y=2.0),
            (3, 2),
        )

        self.assertEqual(
            pipe_jit(eqx.filter_value_and_grad(g))(jnp.array(2.0), y=2.0),
            (5, 2),
        )
        assert num_traces == 2

    def test_partial_jit_vmap(self):
        num_traces = 0

        def f(x, y, z):
            nonlocal num_traces
            num_traces += 1
            return y * x + 1 + z

        out = pipe_jit(make_partial_pipe(pipe_vmap(f))(y=1, z=0))(jnp.array([1, 2]))
        self.assertAllClose(out, jnp.array([2, 3]))
        assert num_traces == 1

        out = pipe_jit(make_partial_pipe(pipe_vmap(f))(x=2, z=0))(jnp.array([2, 3]))
        self.assertAllClose(out, jnp.array([5, 7]))
        assert num_traces == 2


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
