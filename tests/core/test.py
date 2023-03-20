import stamox as stx
import equinox as eqx
import jax.numpy as jnp

import functools as ft

import jax.random as jrandom


def f(x, key):
    return x + jrandom.normal(key, shape=())

def g(x, key):
    return x + jrandom.normal(key, shape=())

def h(x, key):
    return x + jrandom.normal(key, shape=())

f = ft.partial(f, key=jrandom.PRNGKey(0))
g = ft.partial(g, key=jrandom.PRNGKey(0))
h = ft.partial(h, key=jrandom.PRNGKey(0))

f = stx.core.make_pipe(f, name='f')
g = stx.core.make_pipe(g, name='g')
h = stx.core.make_pipe(h, name='h')


pipe = f >> g >> h

