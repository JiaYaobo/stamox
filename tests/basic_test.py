from stamox.core import StateFunc, pipe_jit,  make_partial_pipe, make_pipe, partial_pipe_jit
import jax.random as jrandom
import jax.numpy as jnp
from functools import partial
from jax import jit

from equinox.nn import Linear

from timeit import timeit

a = jnp.ones((10000, ))

# @partial_pipe_jit
# def f(x, y):
#     return x+y

@pipe_jit
def g(x):
    return x ** 2

@make_pipe
def g2(x):
    return x ** 2

h = g2 >> g2
h1 = g2 >> g
h2 = g >> g2
h3 = g >> g


print(timeit(lambda: h(a)))
print(timeit(lambda: h1(a)))
print(timeit(lambda: h2(a)))
print(timeit(lambda: h3(a)))


