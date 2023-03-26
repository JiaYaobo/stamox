import jax.numpy as jnp
import jax.tree_util as jtu
from jax import vmap, jit, lax

from ..util import atleast_1d
from ..maps import auto_map
from ._normal import qnorm

@jit
def _cumm_dgeom(k ,p):

    def cond(carry):
        i, k,  _ = carry
        return i <= k

    def body(carry):
        i, k, ds0 = carry
        ds1 = ds0 + dgeom(i, p)
        i = i + 1
        carry = (i, k, ds1)
        return carry

    i = 0
    init = (i,  k,  jnp.asarray([0.]))
    out = lax.while_loop(cond, body, init)
    return out[2]

def pgeom(k ,p):
    k = jnp.asarray(k ,jnp.int32)
    pp = auto_map(_cumm_dgeom, k, p)
    pp = lax.clamp(0., pp, 1.).ravel()
    return pp


def dgeom(k, p):
    # k = jnp.asarray(k, dtype=jnp.int32)
    pp = auto_map(_dgeom, k, p)
    return pp


@jit
def _dgeom(k, p):
    return jnp.power(1-p, k) * p





