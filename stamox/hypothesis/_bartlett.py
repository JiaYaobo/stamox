"""
In statistics, Bartlett's test, named after Maurice Stevenson Bartlett, is used to test homoscedasticity, 
that is, if multiple samples are from populations with equal variances.Some statistical tests, 
such as the analysis of variance, assume that variances are equal across groups or samples, 
which can be verified with Bartlett's test.
"""

from functools import partial

import jax.numpy as jnp
from equinox import filter_jit
from jax import vmap

from ..core import make_pipe
from ..distribution import pchisq
from ._base import HypoTest


class BartlettTest(HypoTest):
    def __init__(
        self,
        statistic=None,
        parameters=None,
        p_value=None,
        estimate=None,
        null_value=None,
        alternative=None,
    ):
        super().__init__(
            statistic,
            parameters,
            p_value,
            estimate,
            null_value,
            alternative,
            name="BartlettTest",
        )

    @property
    def df(self):
        return self.parameters


@make_pipe
def bartlett_test(*samples) -> BartlettTest:
    """Calculates the Bartlett test statistic for multiple samples.

    Args:
        *samples (array_like): A sequence of 1-D arrays, each containing
            a sample of scores. All samples must have the same length.

    Returns:
        float: The Bartlett test statistic.
    """
    samples = jnp.vstack(samples)
    return _bartlett(samples)


@filter_jit
def _bartlett(samples):
    """Calculates the Bartlett test statistic for multiple samples.

    Args:
        samples (array_like): A 2-D array, each row containing a sample of
            scores. All samples must have the same length.

    Returns:
        float: The Bartlett test statistic.
    """
    k = samples.shape[0]
    Ni = jnp.asarray(vmap(jnp.size, in_axes=(0,))(samples), dtype=jnp.float32)
    ssq = vmap(partial(jnp.var, ddof=1), in_axes=(0,))(samples)
    Ntot = jnp.sum(Ni, axis=0)
    spsq = jnp.sum((Ni - 1) * ssq, axis=0) / (1.0 * (Ntot - k))
    numer = (Ntot * 1.0 - k) * jnp.log(spsq) - jnp.sum(
        (Ni - 1.0) * jnp.log(ssq), axis=0
    )
    denom = 1.0 + 1.0 / (3 * (k - 1)) * (
        (jnp.sum(1.0 / (Ni - 1.0), axis=0)) - 1.0 / (Ntot - k)
    )
    stats = numer / denom
    param = k - 1
    pval = pchisq(stats, param, lower_tail=False)
    return BartlettTest(statistic=stats, parameters=param, p_value=pval)
