from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax._src.random import Shape
from jax.random import KeyArray
from jax.scipy.special import ndtr, ndtri
from jaxtyping import Array, ArrayLike, Bool, Float

from ..core import make_partial_pipe


@filter_jit
def _pnorm(
    x: Union[Float, ArrayLike],
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
):
    scaled = (x - mean) / sd
    return ndtr(scaled)


@make_partial_pipe
def pnorm(
    x: Union[Float, ArrayLike],
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
) -> Array:
    """Calculate the cumulative distribution function (CDF) of the normal distribution.

    Args:
        x (Union[Float, ArrayLike]): The quantiles to calculate the CDF at.
        mean (Union[Float, ArrayLike], optional): The mean of the normal distribution. Defaults to 0.0.
        sd (Union[Float, ArrayLike], optional): The standard deviation of the normal distribution.
            Defaults to 1.0.
        lower_tail (bool, optional): If True, calculate the probability that x is less than or equal to the
            given quantile(s). If False, calculate the probability that x is greater than the given quantile(s).
            Defaults to True.
        log_prob (bool, optional): If True, return the log of the CDF instead of the actual value.
            Defaults to False.

    Returns:
        Union[Float, jnp.ndarray]: The CDF of the normal distribution evaluated at x.


    Examples:
        >>> pnorm(2.0)
        0.977249869052224

        >>> pnorm([1.5, 2.0, 2.5], mean=2.0, sd=0.5, lower_tail=False)
        DeviceArray([0.3085378 , 0.02275013, 0.0013499 ], dtype=float32)

        >>> pnorm([1.5, 2.0, 2.5], mean=2.0, sd=0.5, log_prob=True)
        DeviceArray([-1.177404  , -3.7859507 , -6.5995502 ], dtype=float32)
    """
    x = jnp.atleast_1d(x)
    mean = jnp.asarray(mean)
    sd = jnp.asarray(sd)
    p = filter_vmap(_pnorm)(x, mean, sd)
    if not lower_tail:
        p = 1.0 - p
    if log_prob:
        p = jnp.log(p)
    return p


_dnorm = filter_jit(filter_grad(_pnorm))


@make_partial_pipe
def dnorm(
    x: Union[Float, ArrayLike],
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
) -> Array:
    """
    Probability density function (PDF) for Normal distribution.

    Args:
        x (Union[Float, ArrayLike]): The input value(s) at which to evaluate the PDF.
        mean (Union[Float, ArrayLike], optional): The mean of the normal distribution. Defaults to 0.0.
        sd (Union[Float, ArrayLike], optional): The standard deviation of the normal distribution. Defaults to 1.0.
        lower_tail (bool, optional): If True (default), returns the cumulative distribution function (CDF) from negative infinity up to x. Otherwise, returns the CDF from x to positive infinity.
        log_prob (bool, optional): If True, returns the log-probability instead of the probability.

    Returns:
        grads: The probability density function evaluated at point(s) x.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([0.5, 1.0, -1.5])
        >>> dnorm(x) # doctest: +SKIP
        DeviceArray([0.35206532, 0.24197073, 0.05854983], dtype=float32)
    """
    x = jnp.atleast_1d(x)
    grads = filter_vmap(_dnorm)(x, mean, sd)
    if not lower_tail:
        grads = 1 - grads
    if log_prob:
        grads = jnp.log(grads)
    return grads


@filter_jit
def _qnorm(
    q: Union[Float, ArrayLike],
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
):
    x = ndtri(q)
    return x * sd + mean


@make_partial_pipe
def qnorm(
    q: Union[Float, ArrayLike],
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
) -> Array:
    """
    Calculates the quantile function of the normal distribution for a given probability.

    Args:
        q (float or jnp.ndarray): Probability values.
        mean (float or jnp.ndarray, optional): Mean of the normal distribution. Default is 0.0.
        sd (float or jnp.ndarray, optional): Standard deviation of the normal distribution. Default is 1.0.
        lower_tail (bool, optional): If `True`, returns P(X ≤ x). If `False`, returns P(X > x). Default is `True`.
        log_prob (bool, optional): If `True`, returns the logarithm of the quantile function. Default is `False`.

    Returns:
        jnp.ndarray: The inverse cumulative density function of the normal distribution evaluated at `q`.

    Raises:
        ValueError: When the input probabilities are not between 0 and 1.

    Examples:
        >>> qnorm(0.5)
        array(0.)
        >>> qnorm([0.25, 0.75], mean=3, sd=2)
        array([1.4867225, 4.5132775])
    """
    q = jnp.atleast_1d(q)
    mean = jnp.asarray(mean)
    sd = jnp.asarray(sd)
    if not lower_tail:
        q = 1 - q
    if log_prob:
        q = jnp.exp(q)
    x = filter_vmap(_qnorm)(q, mean, sd)
    return x


@filter_jit
def _rnorm(key, mean, sd, sample_shape):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(mean), jnp.shape(sd))
    mean = jnp.broadcast_to(mean, sample_shape)
    sd = jnp.broadcast_to(sd, sample_shape)
    return jrand.normal(key, sample_shape) * sd + mean


@make_partial_pipe
def rnorm(
    key: KeyArray,
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
    sample_shape: Optional[Shape] = None,
    lower_tail: Bool = True,
    log_prob: Bool = False,
) -> Array:
    """Generates random numbers from a normal distribution.

    Args:
        key: A KeyArray object used to generate the random numbers.
        mean: The mean of the normal distribution. Defaults to 0.0.
        sd: The standard deviation of the normal distribution. Defaults to 1.0.
        sample_shape: An optional tuple of integers specifying the shape of the
        output array. Defaults to an empty tuple.

    Returns:
        A NumPy array containing random numbers from a normal distribution.
    """
    mean = jnp.asarray(mean)
    sd = jnp.asarray(sd)
    probs = _rnorm(key, mean, sd, sample_shape)
    if not lower_tail:
        probs = 1 - probs

    if log_prob:
        probs = jnp.log(probs)

    return probs
