import jax.numpy as jnp
import jax.random as jrand
from jax import jit
from jax.scipy.special import gammainc

from ..maps import auto_map
from ._normal import qnorm

def ppoisson(x, rate):
    p = auto_map(_ppoisson, x, rate)
    return p

@jit
def _ppoisson(x, rate):
    k = jnp.floor(x) + 1.
    return gammainc(k, rate)


def rpoisson(key, rate, sample_shape=()):
    return _rpoisson(key, rate, sample_shape)

def _rpoisson(key, rate, sample_shape=()):
    return jrand.poisson(key, rate, shape=sample_shape)


def qpoisson(q, rate):
    x = auto_map(_qpoisson, q, rate)
    return x

@jit
def _qpoisson(q, rate):
    """
    Computes the p-th quantile of a Poisson distribution with mean lambda_val using the 
    Cornish-Fisher expansion.
    """
    # Compute the z-score corresponding to the desired quantile
    z = qnorm(q)
    
    # Compute the skewness and kurtosis of the Poisson distribution
    skewness = jnp.sqrt(rate)
    kurtosis = 1 / rate
    
    # Compute the third and fourth standardized moments
    standardized_third_moment = skewness**3
    standardized_fourth_moment = 3 + 1 / rate
    
    # Compute the adjusted z-score using the Cornish-Fisher expansion
    adjusted_z = z + (z**2 - 1) * skewness / 6 + (z**3 - 3*z) \
                * (kurtosis - standardized_fourth_moment) / 24 \
                - (2*z**3 - 5*z) * standardized_third_moment**2 / 36
    
    # Compute the approximate quantile using the adjusted z-score
    quantile = rate + jnp.sqrt(rate) * adjusted_z

    return quantile