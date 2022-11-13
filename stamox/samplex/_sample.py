
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
from jax import jit



def sample(key, x, n, replace=True):
    if replace == False:
        raise NotImplementedError("Not Implemented with replace = False")
    size = x.size
    ns =  _sample(key, n, size, replace)
    return x[ns,]


@jtu.Partial(jit, static_argnames=('n', 'replace', ))
def _sample(key, n, size, replace=True):
    return jrand.randint(key, (n, ), 0, size)