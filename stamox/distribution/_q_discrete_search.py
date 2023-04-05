from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax import lax
from jaxtyping import ArrayLike

from ._normal import qnorm


_EPSILON = 1e-15

_pf_n_  = 8    			
_pf_L_  = 2 			
_yLarge_ = 4096	
_incF_ = (1./64)	
_iShrink_ = 8   
_relTol_ = 1e-15			
_xf_ = 4

def do_search(Y_, z, p, incr, lower_tail, log_prob):
    cond1 = z >= p
    left = lax.select(lower_tail, cond1, ~cond1)
    if left:
        pass

    

def q_discr_check_boundary(x):
    return lax.select(x < 0., 0., x)


def q_discrete_body(p, mu, sigma, gamma, lower_tail, log_prob):
    z = qnorm(p, lower_tail=lower_tail, log_prob=log_prob)
    y = lax.round(mu + sigma * (z + gamma * (z * z - 1)))
    y = q_discr_check_boundary(y)

    if log_prob is True:
        e = _pf_L_ * _EPSILON
        if lower_tail is True:
            p = p * (1. + e)
        else:
            p = p * (1. - e)
    else:
        e = _pf_n_ * _EPSILON
        if lower_tail is True:
            p = p * (1 - e)
        else:
            p = lax.select(1 - p > _xf_ * e, p * (1 + e), p)
    



