from typing import Callable

import jax.numpy as jnp
from jax import jit
from jaxtyping import ArrayLike

from ..core import make_partial_pipe


@make_partial_pipe
def step_fun(x, y, ival=0.0, sorted=False, side="left", dtype=jnp.float32):
    """Returns a function that evaluates a step function at given points.

    Args:
        x (array-like): The x-coordinates of the step points.
        y (array-like): The y-coordinates of the step points.
        ival (float, optional): The initial value of the step function. Defaults to 0.
        sorted (bool, optional): Whether the x-coordinates are already sorted. Defaults to False.
        side (str, optional): The side of the interval to take when evaluating the step function. Must be either 'left' or 'right'. Defaults to 'left'.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float32.

    Returns:
        Callable[..., ArrayLike]: A function that evaluates the step function at given points.
    """
    if side.lower() not in ["right", "left"]:
        raise ValueError("side must be left or right")

    _x = jnp.asarray(x, dtype=dtype)
    _y = jnp.asarray(y, dtype=dtype)

    _x = jnp.r_[-jnp.inf, _x]
    _y = jnp.r_[ival, _y]

    if not sorted:
        asort = jnp.argsort(_x)
        _x = jnp.take(_x, asort, 0)
        _y = jnp.take(_y, asort, 0)

    @jit
    def _call(time):
        time = jnp.asarray(time)
        tind = jnp.searchsorted(_x, time, side) - 1
        return _y[tind]

    return _call


@make_partial_pipe
def ecdf(x: ArrayLike, side="right", dtype=jnp.float32) -> Callable[..., ArrayLike]:
    """Calculates the empirical cumulative distribution function (ECDF) of a given array.

    Args:
        x (array): The array to calculate the ECDF for.
        side (str, optional): Specifies which side of the step function to use. Defaults to 'right'.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float32.

    Returns:
        Callable[..., ArrayLike]: A function that evaluates the ECDF at given points.
    """
    x = jnp.array(x, copy=True, dtype=dtype)
    x = jnp.sort(x)
    nobs = x.size
    y = jnp.linspace(1.0 / nobs, 1, nobs)
    return step_fun(x, y, side=side, sorted=True)
