"""
The Friedman test is a non-parametric statistical test developed by Milton Friedman.
Similar to the parametric repeated measures ANOVA, it is used to detect differences in treatments across multiple test attempts. 
The procedure involves ranking each row (or block) together, then considering the values of ranks by columns. 
Applicable to complete block designs, it is thus a special case of the Durbin test.
"""
import jax.numpy as jnp
from jax import vmap, jit


def friedman(*samples):
    samples = jnp.vstack(samples)
    _friedman(samples)

@jit
def _friedman(samples):
    """_summary_
    Supports No tie Only 
    Returns:
        _type_: _description_
    """
    k_treatments, n_blocks = samples.shape
    ranks = vmap(lambda x: jnp.argsort(-x) + 1, in_axes=(1, ))(samples)
    avg_ranks = jnp.mean(ranks, axis=0)
    Q = 12.0 * n_blocks / (k_treatments * (k_treatments + 1) ) * jnp.sum(avg_ranks**2, axis=0) - 3*n_blocks*(k_treatments+1)
    return Q 


