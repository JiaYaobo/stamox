"""Test better partial"""
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from stamox import better_partial


def test_better_partial1():
    def f(a, b, c, d):
        return a + b + c + d

    g = better_partial(f, 1, 2, 3)
    np.testing.assert_allclose(g(4), 10)
    np.testing.assert_allclose(g(5), 11)


def test_better_partial2():
    def f(a, b, c, d):
        return a + b + c + d

    g = better_partial(f, b=1, a=2, c=3)
    np.testing.assert_allclose(g(4), 10)
    np.testing.assert_allclose(g(5), 11)


def test_better_partial3():
    def f(a, b, c, d):
        return a * b + c * d

    g = better_partial(f, b=2, c=3)
    np.testing.assert_allclose(g(1, 2), 8)


def test_better_partial4():
    def f(a, b=None, c=None, d=None):
        return a * b + c * d

    g = better_partial(f, b=2, c=3)
    np.testing.assert_allclose(g(1, 2), 8)


def test_better_partial5():
    def f(a, b, c, d):
        return a * b + c * d

    g = better_partial(f, c=1)
    np.testing.assert_allclose(g(1, 2, 3), 5)


def test_jit_compatible():
    def f(a, b, c, d):
        return a * b + c * d

    f = jit(f)
    g = better_partial(f, 1, 2, 3)
    jit_g = jit(g)
    np.testing.assert_allclose(g(4), 14)
    np.testing.assert_allclose(jit_g(4), 14)


def test_jit_compatible2():
    def f(a, b, c, d):
        return a * b + c * d

    f = jit(f)
    g = better_partial(f, b=1)
    jit_g = jit(g)
    np.testing.assert_allclose(g(2, 3, 4), 14)
    np.testing.assert_allclose(jit_g(2, 3, 4), 14)


def test_vmap_compatible():
    def f(a, b):
        return a * b

    g = better_partial(f, 1)
    vmap_g = vmap(g)
    b = jnp.arange(10)
    np.testing.assert_array_equal(vmap_g(b), b)


def test_vmap_compatible2():
    def f(a, b, c):
        return a * b + c

    g = better_partial(f, b=1)
    vmap_g = vmap(g)
    a = jnp.arange(10)
    c = jnp.arange(10)
    np.testing.assert_array_equal(vmap_g(a, c), a + c)
