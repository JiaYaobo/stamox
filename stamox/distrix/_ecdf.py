import jax.numpy as jnp
from jax import jit


def step_fun(x, y, ival=0., sorted=False, side='left'):
    if side.lower() not in ['right', 'left']:
        raise ValueError("side must be left or right")
    
    _x = jnp.asarray(x)
    _y = jnp.asarray(y)

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


def ecdf(x, side='right'):
    x = jnp.array(x, copy=True)
    x = jnp.sort(x)
    nobs = x.size
    y = jnp.linspace(1./nobs, 1, nobs)
    return step_fun(x, y, side=side, sorted=True)