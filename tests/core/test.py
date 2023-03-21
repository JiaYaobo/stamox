from stamox.core import make_pipe, make_partial_pipe
from jax import grad, vmap, pmap
import jax.random as jrandom
import jax.numpy as jnp

key = jrandom.PRNGKey(1)
x = jrandom.normal(key, (100, ))

@make_partial_pipe
def f(x, exponent):
    return x ** exponent * jnp.sin(x)

@make_partial_pipe
def g(x, aux):
    return x ** 2 + aux
fp = f(exponent=2.)
# autograd 
map_fp = make_pipe(vmap(grad(fp) ,0))
h1 = f(exponent=3.) >> map_fp >> g(aux=0.5)
h2 = map_fp >> g(aux=3.) >> g(aux=1.) >> g(aux=4.)
h = h1 >> h2
h(x)

