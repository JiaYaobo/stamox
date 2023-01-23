import jax.random as jrand
from jax import jit, grad


from ._chisq import rchisq
from ..math.special import fdtri, fdtr
from ..maps import auto_map


def dF(x, dfn, dfd):
    _df = grad(_pf)
    grads = auto_map(_df, x, dfn, dfd)
    return grads


def pF(x, dfn, dfd):
    p = auto_map(_pf, x, dfn, dfd)
    return p


def qF(q, dfn, dfd):
    x = auto_map(_qf, q, dfn, dfd)
    return x


@jit
def _pf(x, dfn, dfd):
    return fdtr(dfn, dfd, x)


@jit
def _qf(q, dfn, dfd):
    return fdtri(dfn, dfd, q)



def rF(key, dfn, dfd, sample_shape=()):
    k1, k2 = jrand.split(key)
    return (rchisq(k1, dfn, sample_shape=sample_shape)/dfn)/(rchisq(k2, dfd, sample_shape=sample_shape)/dfd)
