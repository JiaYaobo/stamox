from functools import wraps
from typing import Callable, TypeVar

import jax.numpy as jnp
from equinox import filter_make_jaxpr
from jax import core, lax
from jax._src.util import safe_map


ReturnValue = TypeVar("ReturnValue")

_inverse_registry = {}

_inverse_registry[lax.exp_p] = jnp.log
_inverse_registry[lax.log_p] = jnp.exp
_inverse_registry[lax.sin_p] = jnp.arcsin
_inverse_registry[lax.cos_p] = jnp.arccos
_inverse_registry[lax.tan_p] = jnp.arctan
_inverse_registry[lax.asin_p] = jnp.sin
_inverse_registry[lax.acos_p] = jnp.cos
_inverse_registry[lax.atan_p] = jnp.tan
_inverse_registry[lax.sinh_p] = jnp.arcsinh
_inverse_registry[lax.cosh_p] = jnp.arccosh
_inverse_registry[lax.tanh_p] = jnp.arctanh
_inverse_registry[lax.asinh_p] = jnp.sinh
_inverse_registry[lax.acosh_p] = jnp.cosh
_inverse_registry[lax.atanh_p] = jnp.tanh


# only support unop now, developing...
def inverse(fun: Callable[..., ReturnValue]) -> Callable[..., ReturnValue]:
    @wraps(fun)
    def wrapped(*args, **kwargs):
        # Since we assume unary functions, we won't worry about flattening and
        # unflattening arguments.
        closed_jaxpr = filter_make_jaxpr(fun)(*args, **kwargs)
        out = inverse_jaxpr(closed_jaxpr[0], closed_jaxpr[1], *args)
        return out[0]
    return wrapped

def inverse_jaxpr(jaxpr, consts, *args):
    env = {}
    
    def read(var):
        if type(var) is core.Literal:
            return var.val
        else:
            return env[var]
    
    def write(var, val):
        env[var] = val
    safe_map(write, jaxpr.outvars, args)
    safe_map(write, jaxpr.constvars, consts)
    # Looping backward
    for eqn in jaxpr.eqns[::-1]:
      #  outvars are now invars 
        invals = safe_map(read, eqn.outvars)
        if eqn.primitive not in _inverse_registry:
            raise NotImplementedError(
            f"{eqn.primitive} does not have registered inverse.")
        # Assuming a unary function 
        outval = _inverse_registry[eqn.primitive](*invals)
        safe_map(write, eqn.invars, [outval])
    return safe_map(read, jaxpr.invars)
        



