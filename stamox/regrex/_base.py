import jax
import jax.numpy as jnp
from jax import jit, vmap

from ._methods import _lm_pinv, _lm_qr


def dispatch(method):
    if method == 'qr':
        return _lm_qr
    elif method == 'pinv':
        return _lm_pinv
    else:
        raise ValueError('Only support qr and pinv now')
