import jax.numpy as jnp
from jax import vmap, jit

from ..distrix import pnorm, pt


@jit
def cor2p(cor, n, method="pearson"):
    # Statistics
    if method == "kendall":
        stats = (3 * cor * jnp.sqrt(n * (n-1))) / jnp.sqrt(2 * (2 * n + 5))
    else:
        stats = cor * jnp.sqrt((n - 2) / (1 - cor ** 2))

    # p value
    if method == "kendall":
        p = 2 * pnorm(-jnp.abs(stats))
    else:
        p = 2 * pt(-jnp.abs(stats), df=n - 2)

    return stats, p

    
