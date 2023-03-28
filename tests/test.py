from stamox.core import make_pipe, make_partial_pipe, Pipeable, pipe_vmap

import jax.numpy as jnp
from equinox import filter_grad, filter_vmap

x = jnp.ones((1000, ))

@make_pipe
def f(x):
    return x + 1

@make_partial_pipe
def g(x, y):
    return x ** 2 + y

@pipe_vmap
@filter_grad
def m(x):
    return x * 2


z = Pipeable(x) >> f >> g(y=1.) >> m >> g(y=2.) >> f

print(z())
