"""Test for F distribution"""
import jax.random as jrand
import numpy as np
from scipy.stats import f

import stamox.pipe_functions as PF
from stamox.distribution import dF, pF, qF, rF


np.random.seed(1)


def test_rF():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    dfn = 5.0
    dfd = 5.0
    rvs = rF(key, sample_shape, dfn, dfd)
    avg = rvs.mean()
    np.testing.assert_allclose(avg, dfd / (dfd - 2), atol=1e-2)


def test_pF():
    x = np.random.f(5, 5, 1000)
    dfn = 5.0
    dfd = 5.0
    p = pF(x, dfn, dfd)
    true_p = f.cdf(x, dfn, dfd)
    np.testing.assert_allclose(p, true_p, atol=1e-5)


def test_qF():
    q = np.random.uniform(0, 1.0, 1000)
    dfn = 5.0
    dfd = 5.0
    x = qF(q, dfn, dfd, dtype=np.float64)
    true_x = f.ppf(q, dfn, dfd)
    np.testing.assert_allclose(x, true_x)


def test_dF():
    x = np.random.f(5, 5, 1000)
    dfn = 5.0
    dfd = 5.0
    grads = dF(x, dfn, dfd, dtype=np.float64)
    true_grads = f.pdf(x, dfn, dfd)
    np.testing.assert_allclose(grads, true_grads)


def test_partial_pF():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    dfn = 5.0
    dfd = 5.0
    p = PF.pF(dfn=dfn, dfd=dfd)(x)
    true_p = np.array([0.01224192, 0.05096974, 0.10620910, 0.16868416, 0.23251132])
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_partial_dF():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    dfn = 5.0
    dfd = 5.0
    grads = PF.dF(dfn=dfn, dfd=dfd)(x)
    true_grads = np.array([0.2666708, 0.4881773, 0.6010408, 0.6388349, 0.6323209])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_qF():
    q = np.array([0.01224192, 0.05096974, 0.10620910, 0.16868416, 0.23251132])
    dfn = 5.0
    dfd = 5.0
    x = PF.qF(dfn=dfn, dfd=dfd)(q)
    true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(x, true_x, atol=1e-4)


def test_partial_rF():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    dfn = 5.0
    dfd = 5.0
    rvs = PF.rF(dfn=dfn, dfd=dfd, sample_shape=sample_shape)(key)
    avg = rvs.mean()
    np.testing.assert_allclose(avg, dfd / (dfd - 2), atol=1e-2)
