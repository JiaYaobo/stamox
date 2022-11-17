import jax.numpy as jnp

from jax import jit, grad, lax, vmap
import jax.tree_util as jtu


def scad(a, tau, c):
    a = jnp.asarray(a)
    a = jnp.atleast_1d(a)
    res = vmap(_scad, in_axes=(0, None, None))(a, tau, c)
    return res


def dscad(a, tau, c):
    a = jnp.asarray(a)
    a = jnp.atleast_1d(a)
    res = vmap(jit(grad(_scad)), in_axes=(0, None, None))(a, tau, c)
    return res


def d2scad(a, tau, c):
    a = jnp.asarray(a)
    a = jnp.atleast_1d(a)
    res = vmap(jit(grad(jit(grad(_scad)))), in_axes=(0, None, None))(a, tau, c)
    return res


@jtu.Partial(jit, static_argnames=('tau', 'c', ))
def _scad(a, tau, c):
    a_abs = jnp.abs(a)
    mask1 = a_abs < tau
    mask3 = a_abs >= c * tau
    mask2 = ~mask1 & ~mask3
    branches = [lambda x: tau * x, 
                lambda x:  x ** 2 - 2*c * tau * x + tau ** 2, 
                lambda x: (c + 1) * tau ** 2 / 2.]
    index = jnp.argwhere(jnp.array([mask1, mask2, mask2]), size=1).squeeze()
    res = lax.switch(index, branches, a_abs)
    return res
    

@jtu.Partial(jit, static_argnames=('weights', 'alpha', ))
def l2(a, weights=1., alpha=1.):
    return jnp.sum(weights * alpha * a ** 2)


@jtu.Partial(jit, static_argnames=('weights', 'alpha', ))
def dl2(a, weights=1., alpha=1.):
    return jit(grad(l2))(a, weights, alpha)


@jtu.Partial(jit, static_argnames=('weights', 'alpha', ))
def d2l2(a, weights=1., alpha=1.):
    return jit(grad(jit(grad(l2))))(a, weights, alpha)
