import jax.numpy as jnp
from jax import vmap

def cube_map(func, *inputs):

    n = len(inputs)
    args = []
    arg_sizes = []
    all_scalar = True


    for inp in inputs:
        if not jnp.isscalar(inp):
            all_scalar = False
        
        inp = jnp.asarray(inp)
        args.append(inp)
        arg_sizes.append(inp.size)

    
    if all_scalar:
        return jnp.atleast_1d(func(*args))

    else:
        for i in range(n):
            in_axes = [None] * n
            if arg_sizes[n-i-1] > 1:
                in_axes[n-i-1] = 0
                func = vmap(func, in_axes=in_axes)
    

    return func(*args)
    

