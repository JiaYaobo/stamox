from functools import partial

import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import filter_jit
from jax import Array, jit

from ...distribution import pF
from ...hypothesis import HypoTest


class OneWayAnovaTest(HypoTest):
    df_between: int
    df_within: int
    df_tol: int
    ss_between: float
    ss_within: float
    ss_total: float
    ms_between: float
    ms_within: float

    def __init__(
        self,
        statistic=None,
        p_value=None,
        df_between=None,
        df_within=None,
        df_total=None,
        ss_between=None,
        ss_within=None,
        ss_total=None,
        ms_between=None,
        ms_within=None,
        name="One Way Anova Test",
    ):
        super().__init__(statistic=statistic, p_value=p_value, name=name)
        self.df_between = df_between
        self.df_within = df_within
        self.df_tol = df_total
        self.ss_between = ss_between
        self.ss_within = ss_within
        self.ss_total = ss_total
        self.ms_between = ms_between
        self.ms_within = ms_within

    def _summary(self):
        raise NotImplementedError


@partial(jit, static_argnames=("axis"))
def _square_of_sums(a, axis=0) -> Array:
    s = jnp.sum(a, axis, keepdims=True)
    return s * s


@partial(jit, static_argnames=("axis"))
def _sum_of_squares(a, axis=0) -> Array:
    return jnp.sum(a * a, axis=axis, keepdims=True)


def one_way(*samples, axis=0) -> OneWayAnovaTest:
    """Performs a one-way ANOVA test.

    Args:
        *samples: A sequence of samples to compare.
        axis (int): The axis along which the samples are compared.

    Returns:
        OneWayAnovaTest: The result of the one-way ANOVA test.

    Example:
        >>> import jax.numpy as jnp
        >>> from stamox.functions import one_way
        >>>  # 3 samples with 5 observations each
        >>> a = jnp.array([1, 2, 3, 4, 5])
        >>> b = jnp.array([2, 3, 4, 5, 6])
        >>> c = jnp.array([3, 4, 5, 6, 7])
        >>> one_way(a, b, c)

    """
    samples = [jnp.asarray(sample) for sample in samples]
    ngroups = len(samples)
    return _one_way(samples, ngroups, axis=axis)


@filter_jit
def _one_way(samples, ngroups, axis=0):
    alldata = jnp.concatenate(samples, axis=axis)
    N = alldata.shape[axis]
    offset = jnp.mean(alldata, axis=axis, keepdims=True)
    alldata -= offset
    normalized_ss = _square_of_sums(alldata, axis=axis) / N

    # total
    sstot = _sum_of_squares(alldata, axis=axis) - normalized_ss

    # between groups or namely treatments
    ssbn = jnp.sum(
        jnp.concatenate(
            jtu.tree_map(
                lambda s: _square_of_sums(s - offset, axis) / s.shape[0], samples
            ),
            axis,
        ),
        axis,
        keepdims=True,
    )
    ssbn -= normalized_ss

    # with in groups
    sswn = sstot - ssbn

    # degree of freedom
    dfbn = ngroups - 1
    dfwn = N - ngroups

    # mean squares
    msb = ssbn / dfbn
    msw = sswn / dfwn

    # F-ratio
    f = msb / msw
    f = f.squeeze(axis=axis)
    prob = 1 - pF(f, dfbn, dfwn)
    return OneWayAnovaTest(
        statistic=f,
        p_value=prob,
        df_between=dfbn,
        df_within=dfwn,
        df_total=N - 1,
        ss_between=ssbn[0],
        ss_within=sswn[0],
        ss_total=sstot[0],
        ms_between=msb[0],
        ms_within=msw[0],
    )
