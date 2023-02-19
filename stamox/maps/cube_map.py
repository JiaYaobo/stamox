import jax.numpy as jnp
from jax import vmap

def cube_map(func, *inputs):

    n = len(inputs)
    if n == 1:
        return vmap(func, in_axes=0)(*inputs)
    args = []
    for inp in inputs:
        inp = jnp.asarray(inp)
        args.append(inp)

    for i in range(n):
        in_axes = [None] * n
        in_axes[n-i-1] = 0
        func = vmap(func, in_axes=in_axes)
    

    return func(*args)
    

