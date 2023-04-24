from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrand
from equinox import filter_grad, filter_jit, filter_vmap
from jax import lax
from jax._src.random import Shape
from jax.random import KeyArray
from jax.scipy.special import ndtr, ndtri
from jaxtyping import ArrayLike, Bool, Float

from ._utils import (
    _check_clip_distribution_domain,
    _check_clip_probability,
    _post_process,
    _promote_dtype_to_floating,
)


@filter_jit
def _pnorm(
    x: Union[Float, ArrayLike],
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
):
    mean = jnp.asarray(mean, dtype=x.dtype)
    sd = jnp.asarray(sd, dtype=x.dtype)
    scaled = lax.div(lax.sub(x, mean), sd)
    return ndtr(scaled)


def pnorm(
    q: Union[Float, ArrayLike],
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Calculate the cumulative distribution function (CDF) of the normal distribution.

    Args:
        q (Union[Float, ArrayLike]): The quantiles to calculate the CDF at.
        mean (Union[Float, ArrayLike], optional): The mean of the normal distribution. Defaults to 0.0.
        sd (Union[Float, ArrayLike], optional): The standard deviation of the normal distribution.
            Defaults to 1.0.
        lower_tail (bool, optional): If True, calculate the probability that x is less than or equal to the
            given quantile(s). If False, calculate the probability that x is greater than the given quantile(s).
            Defaults to True.
        log_prob (bool, optional): If True, return the log of the CDF instead of the actual value.
            Defaults to False.
        dtype (jnp.dtype, optional): The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: The CDF of the normal distribution evaluated at x.


    Examples:
        >>> pnorm(2.0)
        Array([0.97724986], dtype=float32)

        >>> pnorm([1.5, 2.0, 2.5], mean=2.0, sd=0.5, lower_tail=False)
        Array([0.8413447 , 0.5       , 0.15865529], dtype=float32)
    """
    q, _ = _promote_dtype_to_floating(q, dtype)
    q = jnp.atleast_1d(q)
    q = _check_clip_distribution_domain(q)
    p = filter_vmap(_pnorm)(q, mean, sd)
    p = _post_process(p, lower_tail, log_prob)
    return p


_dnorm = filter_jit(filter_grad(_pnorm))


def dnorm(
    x: Union[Float, ArrayLike],
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """
    Probability density function (PDF) for Normal distribution.

    Args:
        x (Union[Float, ArrayLike]): The input value(s) at which to evaluate the PDF.
        mean (Union[Float, ArrayLike], optional): The mean of the normal distribution. Defaults to 0.0.
        sd (Union[Float, ArrayLike], optional): The standard deviation of the normal distribution. Defaults to 1.0.
        lower_tail (bool, optional): If True (default), returns the cumulative distribution function (CDF) from negative infinity up to x. Otherwise, returns the CDF from x to positive infinity.
        log_prob (bool, optional): If True, returns the log-probability instead of the probability.
        dtype (jnp.dtype, optional): The dtype of the output. Default is `jnp.float_`.

    Returns:
        ArrayLike: The probability density function evaluated at point(s) x.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([0.5, 1.0, -1.5])
        >>> dnorm(x)
        Array([0.35206532, 0.24197075, 0.12951761], dtype=float32)
    """
    x, _ = _promote_dtype_to_floating(x, dtype)
    x = jnp.atleast_1d(x)
    x = _check_clip_distribution_domain(x)
    grads = filter_vmap(_dnorm)(x, mean, sd)
    grads = _post_process(grads, lower_tail, log_prob)
    return grads


@filter_jit
def _qnorm(
    p: Union[Float, ArrayLike],
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
):
    x = ndtri(p)
    sd = lax.convert_element_type(sd, x.dtype)
    mean = lax.convert_element_type(mean, x.dtype)
    return lax.add(lax.mul(x, sd), mean)


def qnorm(
    p: Union[Float, ArrayLike],
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
    lower_tail=True,
    log_prob=False,
    dtype=jnp.float_,
) -> ArrayLike:
    """
    Calculates the quantile function of the normal distribution for a given probability.

    Args:
        p (float or jnp.ndarray): Probability values.
        mean (float or jnp.ndarray, optional): Mean of the normal distribution. Default is 0.0.
        sd (float or jnp.ndarray, optional): Standard deviation of the normal distribution. Default is 1.0.
        lower_tail (bool, optional): If `True`, returns P(X ≤ x). If `False`, returns P(X > x). Default is `True`.
        log_prob (bool, optional): If `True`, returns the logarithm of the quantile function. Default is `False`.
        dtype (jnp.dtype, optional): The dtype of the output. Default is `jnp.float_`.

    Returns:
        ArrayLike: The inverse cumulative density function of the normal distribution evaluated at `q`.


    Examples:
        >>> qnorm(0.5)
        Array([0.], dtype=float32)
        >>> qnorm([0.25, 0.75], mean=3, sd=2)
        Array([1.6510204, 4.3489795], dtype=float32)
    """
    p, _ = _promote_dtype_to_floating(p, dtype)
    p = jnp.atleast_1d(p)
    p = _check_clip_probability(p, lower_tail, log_prob)
    x = filter_vmap(_qnorm)(p, mean, sd)
    return x


@filter_jit
def _rnorm(key, mean, sd, sample_shape, dtype):
    if sample_shape is None:
        sample_shape = jnp.broadcast_shapes(jnp.shape(mean), jnp.shape(sd))
    mean = jnp.broadcast_to(mean, sample_shape)
    sd = jnp.broadcast_to(sd, sample_shape)
    return jrand.normal(key, sample_shape, dtype=dtype) * sd + mean


def rnorm(
    key: KeyArray,
    sample_shape: Optional[Shape] = None,
    mean: Union[Float, ArrayLike] = 0.0,
    sd: Union[Float, ArrayLike] = 1.0,
    lower_tail: Bool = True,
    log_prob: Bool = False,
    dtype=jnp.float_,
) -> ArrayLike:
    """Generates random variables from a normal distribution.

    Args:
        key: A KeyArray object used to generate the random numbers.
        sample_shape: An optional tuple of integers specifying the shape of the
        output array. Defaults to an empty tuple.
        mean: The mean of the normal distribution. Defaults to 0.0.
        sd: The standard deviation of the normal distribution. Defaults to 1.0.
        lower_tail: If True (default), returns the cumulative distribution function (CDF) from negative infinity up to x. Otherwise, returns the CDF from x to positive infinity.
        log_prob: If True, returns the log-probability instead of the probability.
        dtype: The dtype of the output. Defaults to jnp.float_.

    Returns:
        ArrayLike: Random samples from a normal distribution.

    Example:
        >>> key = random.PRNGKey(0)
        >>> rnorm(key, sample_shape=(3, 2))
        Array([[ 0.18784384, -1.2833426 ],
                [ 0.6494181 ,  1.2490594 ],
                [ 0.24447003, -0.11744965]], dtype=float32)

    """
    rvs = _rnorm(key, mean, sd, sample_shape, dtype=dtype)
    if not lower_tail:
        rvs = 1 - rvs

    if log_prob:
        rvs = jnp.log(rvs)

    return rvs
