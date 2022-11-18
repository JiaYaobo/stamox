import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
from jax import vmap, jit, lax

from ..util import zero_dim_to_1_dim_array

@jtu.Partial(jit, static_argnames=('p', ))
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
    pp = vmap(_cumm_dgeom, in_axes=(0, None))(k, p)
    pp = lax.clamp(0., pp, 1.)
    pp = jnp.squeeze(pp, axis=1)
    return pp


def dgeom(k, p):
    k = jnp.asarray(k, dtype=jnp.int32)
    k = zero_dim_to_1_dim_array(k)
    pp = vmap(_dgeom, in_axes=(0, None))(k, p)
    return pp


@jtu.Partial(jit, static_argnames=('p', ))
def _dgeom(k, p):
    return jnp.power(1-p, k - 1) * p


