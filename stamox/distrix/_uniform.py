import jax.numpy as jnp
import jax.random as jrand
from jax import jit

from ..util import zero_dim_to_1_dim_array
from ..maps import auto_map


def dunif(x, mini=0., maxi=1.):
    x = jnp.asarray(x)
    x = zero_dim_to_1_dim_array(x)
    return 1/(maxi - mini) * jnp.ones_like(x)

def punif(x, mini=0., maxi=1.):
    p = auto_map(_punif, x, mini, maxi)
    return p


@jit
def _punif(x, mini=0., maxi=1.):
    p = (x - mini) / (maxi - mini)
    return p


def qunif(q,  mini=0., maxi=1.):
    x = auto_map(_qunif, q, mini, maxi)
    return x


@jit
def _qunif(q, mini=0., maxi=1.):
    x = q * (maxi - mini) + mini
    return x


def runif(key, mini=0., maxi=1., sample_shape=()):
    return _runif(key, mini, maxi, sample_shape)


def _runif(key, mini, maxi, sample_shape=()):
    return jrand.uniform(key, sample_shape, minval=mini, maxval=maxi)
