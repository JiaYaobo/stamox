import jax.numpy as jnp
from jax import vmap
from ..util import atleast_1d


def align_inputs(*inputs):
    max_size = 1
    n = len(inputs)

    args = []
    arg_sizes = []


    if n == 1:
        arg = atleast_1d(inputs[0])
        return arg, [arg.size]

    for arg in inputs:
        arg = jnp.asarray(arg)
        arg_sizes.append(arg.size)
        args.append(arg)
        if arg.size > max_size:
            max_size = arg.size

    for i in range(n):
        if args[i].size == 1 or args[i].ndim == max_size:
            continue
        else:
            args[i] = jnp.tile(
                args[i], 1+(max_size // args[i].size))[:max_size]
            arg_sizes[i] = max_size

    return args, arg_sizes


def auto_map(func, *inputs):

    n = len(inputs)
        
    args, arg_sizes = align_inputs(*inputs)


    if n == 1:
        return vmap(func, in_axes=0)(args)
        
    in_axes = []
    for arg_size in arg_sizes:
        if arg_size > 1:
            in_axes.append(0)
        else:
            in_axes.append(None)
    return vmap(func, in_axes=in_axes)(*args)
