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


def qgeom(q, p):
    x = auto_map(_qgeom, q, p)
    return x

@jit
def _qgeom(q, p):
    """Compute the alpha-quantile of a geometric distribution 
    using Cornish-Fisher Expansion."""
    q = qnorm(q) # the alpha-quantile of the standard normal distribution
    mu = 1 / p # the mean of the geometric distribution
    var = (1 - p) / p**2 # the variance of the geometric distribution
    skew = (2 - p) / jnp.sqrt(1 - p) # the skewness of the geometric distribution
    kurt = 6 + (p**2 - 6 * p + 6) / (1 - p)
    z = q + (q**2 - 1) / 6 * skew + (q**3 - 3 * q) / 24 \
        * kurt - (q**3 - q) / 36 * skew**2
    x = mu + z * jnp.sqrt(var)
    return x


