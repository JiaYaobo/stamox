"""Test for chisqonential distribution"""
import jax.random as jrand
import numpy as np
from scipy.stats import chi2

import stamox.pipe_functions as PF
from stamox.distribution import dchisq, pchisq, qchisq, rchisq


np.random.seed(1)


def test_rchisq():
    key = jrand.PRNGKey(19751002)
    sample_shape = (10000000,)
    df = 3.0
    ts = rchisq(key, sample_shape, df)
    avg = ts.mean()
    var = ts.var(ddof=1)
    np.testing.assert_allclose(avg, df, atol=1e-2)
    np.testing.assert_allclose(var, 2 * df, atol=1e-2)


def test_pchisq():
    x = np.random.chisquare(3, 1000)
    df = 3
    p = pchisq(x, df, dtype=np.float64)
    true_p = chi2(df).cdf(x)
    np.testing.assert_allclose(p, true_p)


def test_qchisq():
    q = np.random.uniform(0, 0.999, 1000)
    df = 3
    x = qchisq(q, df, dtype=np.float64)
    true_x = chi2(df).ppf(q)
    np.testing.assert_allclose(x, true_x)


def test_dchisq():
    x = np.random.chisquare(3, 1000)
    df = 3.0
    grads = dchisq(x, df, dtype=np.float64)
    true_grads = chi2(df).pdf(x)
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_pchisq():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    df = 3
    p = PF.pchisq(df=df, lower_tail=False)(x)
    true_p = np.array([0.991837424, 0.977589298, 0.960028480, 0.940242495, 0.918891412])
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_partial_qchisq():
    q = 1 - np.array([0.008162576, 0.022410702, 0.039971520, 0.059757505, 0.081108588])
    df = 3
    x = PF.qchisq(df=df, lower_tail=False, dtype=np.float64)(q)
    true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(x, true_x, atol=1e-6)


def test_partial_dchisq():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    df = 3.0
    grads = PF.dchisq(df=df)(x)
    true_grads = np.array([0.1200039, 0.1614342, 0.1880730, 0.2065766, 0.2196956])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_rchisq():
    key = jrand.PRNGKey(19751002)
    sample_shape = (10000000,)
    df = 3.0
    ts = PF.rchisq(df=df)(key, sample_shape=sample_shape)
    avg = ts.mean()
    var = ts.var(ddof=1)
    np.testing.assert_allclose(avg, df, atol=1e-2)
    np.testing.assert_allclose(var, 2 * df, atol=1e-2)
