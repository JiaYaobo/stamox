from functools import partial

import jax.numpy as jnp
from jax  import jit


@partial(jit, static_argnames=('lower_tail', 'log_p'))
def logp_or_lower_tail(probs, lower_tail=True, log_p = False):

    if not lower_tail:
        probs = 1 - probs
    if log_p:
        probs = jnp.log(probs)

    return probs