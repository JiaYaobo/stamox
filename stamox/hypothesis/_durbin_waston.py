import jax.numpy as jnp
from equinox import filter_jit


from ._base import HypoTest
from ..core import make_partial_pipe


class DurbinWastonTest(HypoTest):
    def __init__(
        self,
        statistic=None,
        parameters=None,
        p_value=None,
        estimate=None,
        null_value=None,
        alternative=None,
        name="Durbin-Waston Test",
    ):
        super().__init__(
            statistic, parameters, p_value, estimate, null_value, alternative, name
        )


@make_partial_pipe
def durbin_waston(resids, axis=0):
    """Computes the Durbin-Watson statistic for a given array of residuals.

    Args:
        resids (array): An array of residuals.
        axis (int, optional): The axis along which to compute the statistic. Defaults to 0.

    Returns:
        float: The Durbin-Watson statistic.
    """
    resids = jnp.atleast_1d(resids)
    return _durbin_watson(resids, axis)


@filter_jit
def _durbin_watson(resids, axis=0):
    """Computes the Durbin-Watson statistic for a given array of residuals.

    Args:
        resids (array): An array of residuals.
        axis (int, optional): The axis along which to compute the statistic. Defaults to 0.

    Returns:
        float: The Durbin-Watson statistic.
    """
    diff_resids = jnp.diff(resids, 1, axis=axis)
    dw = jnp.sum(diff_resids**2, axis=axis) / jnp.sum(resids**2, axis=axis)
    return DurbinWastonTest(statistic=dw)
