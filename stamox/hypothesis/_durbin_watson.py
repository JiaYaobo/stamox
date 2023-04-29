import jax.numpy as jnp
from equinox import filter_jit
from jaxtyping import ArrayLike

from ..core import make_partial_pipe
from ._base import HypoTest


class DurbinWatsonTest(HypoTest):
    """
    Class for performing the Durbin-Watson Test.

    This class is a subclass of HypoTest and provides methods to perform the Durbin-Waston Test.

    Attributes:
        statistic (float): The test statistic.
        parameters (tuple): The parameters of the test.
        p_value (float): The p-value of the test.
    """

    def __init__(
        self,
        statistic=None,
        parameters=None,
        p_value=None,
        estimate=None,
        null_value=None,
        alternative=None,
        name="Durbin-Watson Test",
    ):
        super().__init__(
            statistic, parameters, p_value, estimate, null_value, alternative, name
        )

    def __repr__(self):
        return f"{self.name}(statistic={self.statistic}, parameters={self.parameters}, p_value={self.p_value})"


@make_partial_pipe
def durbin_watson_test(resids: ArrayLike, axis: int = 0):
    """Computes the Durbin-Watson statistic for a given array of residuals.

    Args:
        resids (array): An array of residuals.
        axis (int, optional): The axis along which to compute the statistic. Defaults to 0.

    Returns:
        DurbinWatsonTest: The Durbin-Watson Test object.

    Example:
        >>> import jax.numpy as jnp
        >>> from stamox.functions import durbin_watson_test
        >>> resids = jnp.array([1, 2, 3, 4, 5])
        >>> durbin_watson_test(resids)
        Durbin-Waston Test(statistic=0.0, parameters=None, p_value=None)
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
    dw = jnp.sum(diff_resids**2, axis=axis, keepdims=True) / jnp.sum(
        resids**2, axis=axis, keepdims=True
    )
    dw = dw.squeeze()
    return DurbinWatsonTest(statistic=dw)
