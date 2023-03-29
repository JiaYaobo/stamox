"""
The Friedman test is a non-parametric statistical test developed by Milton Friedman.
Similar to the parametric repeated measures ANOVA, it is used to detect differences in treatments across multiple test attempts. 
The procedure involves ranking each row (or block) together, then considering the values of ranks by columns. 
Applicable to complete block designs, it is thus a special case of the Durbin test.
"""
import jax.numpy as jnp
from equinox import filter_jit
from jax import vmap
from jax.scipy.stats import rankdata

from ..core import make_pipe
from ..distribution import pchisq
from ._base import HypoTest


class FriedmanTest(HypoTest):
    def __init__(
        self,
        statistic=None,
        parameters=None,
        p_value=None,
        estimate=None,
        null_value=None,
        alternative=None,
        name="Friedman Rank Sum Test",
    ):
        super().__init__(
            statistic, parameters, p_value, estimate, null_value, alternative, name
        )


@make_pipe
def friedman_test(*samples) -> FriedmanTest:
    """Computes the Friedman statistic for a set of samples.

    Args:
        *samples: A sequence of samples, each sample being a sequence of
            observations.

    Returns:
        The computed Friedman statistic.
    """
    ImportWarning("This function is not yet functioned with ties. Use with caution.")
    samples = jnp.vstack(samples).T
    return _friedman(samples)


@filter_jit
def _friedman(samples):
    """Computes the Friedman statistic for a set of samples.

    Args:
        samples: A 2D array of samples, with the first dimension representing
            the treatments and the second dimension representing the blocks.

    Returns:
        The computed Friedman statistic.
    """
    n_blocks, k_treatments = samples.shape
    ranks = vmap(lambda x: rankdata(x))(samples)
    avg_ranks = jnp.mean(ranks, axis=0)
    Q = 12.0 * n_blocks / (k_treatments * (k_treatments + 1)) * jnp.sum(
        avg_ranks**2, axis=0, keepdims=True
    ) - 3 * n_blocks * (k_treatments + 1)
    param = k_treatments - 1
    pval = pchisq(Q, param, lower_tail=False)
    return FriedmanTest(statistic=Q, parameters=param, p_value=pval)
