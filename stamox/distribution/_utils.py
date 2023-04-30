import functools

import jax.numpy as jnp
import numpy as np
from equinox import filter_vmap
from jax import jit, lax


inf = jnp.inf
nan = jnp.nan


@functools.partial(jit, static_argnums=(1, 2))
def _check_clip_probability(p, lower_tail=True, log_prob=False):
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.exp(p)
    return jnp.clip(p, 0.0, 1.0)


@functools.partial(jit, static_argnums=(1, 2))
def _check_clip_distribution_domain(x, lower=-inf, upper=inf):
    return jnp.clip(x, lower, upper)


def _promote_dtype_to_floating(q, dtype):
    if dtype is None:
        if jnp.issubdtype(lax.dtype(q), jnp.integer):
            return jnp.asarray(q, dtype=jnp.float_), jnp.float_
        else:
            return jnp.asarray(q), lax.dtype(q)
    else:
        return jnp.asarray(q, dtype=dtype), dtype


def _promote_dtype_to_integer(x, dtype):
    if dtype is None:
        if jnp.issubdtype(lax.dtype(x), jnp.floating):
            return jnp.asarray(x, dtype=jnp.int_), jnp.int_
        else:
            return jnp.asarray(x), lax.dtype(x)
    else:
        return jnp.asarray(x, dtype=dtype), dtype


@functools.partial(jit, static_argnums=(1, 2))
def _post_process(p, lower_tail=True, log_prob=False):
    if not lower_tail:
        p = 1 - p
    if log_prob:
        p = jnp.log(p)
    return p


def _check_all_scalar(*args):
    for arg in args:
        if not jnp.shape(arg) == ():
            return False
    return True


def svmap_(f, *args):
    if _check_all_scalar(*args):
        return f(*args)
    else:
        return filter_vmap(f)(*args)
    
def _flatten_shapes(*args):
    shape = jnp.broadcast_shapes(*[jnp.shape(arg) for arg in args])
    ravel_shape = (np.prod(shape),)
    return shape, ravel_shape


__all__ = [  # noqa: F405
    "inf",
    "nan",
    "_check_clip_probability",
    "_check_clip_distribution_domain",
    "_promote_dtype_to_floating",
    "_promote_dtype_to_integer",
    "_post_process",
    "_check_all_scalar",
    "svmap_",
    "_flatten_shapes",
]
