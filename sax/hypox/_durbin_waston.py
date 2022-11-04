import functools as ft

import jax.numpy as jnp
from jax import vmap, jit

from sax.util import zero_dim_to_1_dim_array


def durbin_waston(resids, axis=0):
    resids = jnp.asarray(resids)
    axis = jnp.asarray(axis)
    resids = zero_dim_to_1_dim_array(resids)
    dws = vmap(_durbin_watson, in_axes=(0, None))(resids, axis)
    return dws

@ft.partial(jit, static_argnames=('axis', ))
def _durbin_watson(resids, axis=0):
    r"""
    Calculates the Durbin-Watson statistic.

    Parameters
    ----------
    resids : array_like
        Data for which to compute the Durbin-Watson statistic. Usually
        regression model residuals.
    axis : int, optional
        Axis to use if data has more than 1 dimension. Default is 0.

    Returns
    -------
    dw : float, array_like
        The Durbin-Watson statistic.

    Notes
    -----
    The null hypothesis of the test is that there is no serial correlation
    in the residuals.
    The Durbin-Watson test statistic is defined as:

    .. math::

       \sum_{t=2}^T((e_t - e_{t-1})^2)/\sum_{t=1}^Te_t^2

    The test statistic is approximately equal to 2*(1-r) where ``r`` is the
    sample autocorrelation of the residuals. Thus, for r == 0, indicating no
    serial correlation, the test statistic equals 2. This statistic will
    always be between 0 and 4. The closer to 0 the statistic, the more
    evidence for positive serial correlation. The closer to 4, the more
    evidence for negative serial correlation.
    """
    resids = jnp.asarray(resids)
    diff_resids = jnp.diff(resids, 1, axis=axis)
    dw = jnp.sum(diff_resids**2, axis=axis) / jnp.sum(resids**2, axis=axis)
    return dw

