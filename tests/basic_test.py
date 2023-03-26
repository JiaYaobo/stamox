import jax.numpy as jnp
import jax
import jax.random as jrandom
from functools import partial
from equinox import filter_jit, filter_vmap, filter_grad, filter_make_jaxpr
from stamox.basic import mean
from stamox.core import  make_pipe, make_partial_pipe


@make_partial_pipe
def f(q, aux):
    return q ** 3 + aux

a = jnp.array([1., 2.])


g = f(aux=1.) >> jnp.mean

print(g)