"""Test for poisson distribution"""
import jax.random as jrand
import numpy as np
from scipy.stats import poisson

import stamox.pipe_functions as PF
from stamox.distribution import dpoisson, ppoisson, qpoisson, rpoisson


def test_rpoisson():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    rate = 2.5
    ts = rpoisson(key, sample_shape, rate)
    avg = ts.mean()
    var = ts.var(ddof=1)
    np.testing.assert_allclose(avg, rate, atol=1e-2)
    np.testing.assert_allclose(var, rate, atol=1e-2)


def test_ppoisson():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rate = 2.5
    p = ppoisson(x, rate)
    true_p = np.array([0.2872975, 0.5438131, 0.7575761, 0.8911780, 0.9579790])
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_dpoisson():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rate = 2.5
    grads = dpoisson(x, rate)
    true_grads = np.array([0.20521250, 0.25651562, 0.21376302, 0.13360189, 0.06680094])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_ppoisson():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rate = 2.5
    p = PF.ppoisson(rate=rate, lower_tail=False)(x)
    true_p = np.array([0.7127025, 0.4561869, 0.2424239, 0.1088220, 0.0420210])
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_qpoisson():
    p = np.array([0.1, 0.5, 0.9])
    rate = 2.5
    q = qpoisson(p, rate)
    true_q = poisson.ppf(p, rate)
    np.testing.assert_allclose(q, true_q, atol=1e-6)


def test_partial_dpoisson():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rate = 2.5
    grads = PF.dpoisson(rate=rate)(x)
    true_grads = np.array([0.20521250, 0.25651562, 0.21376302, 0.13360189, 0.06680094])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_rpoisson():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    rate = 2.5
    ts = PF.rpoisson(rate=rate)(key, sample_shape=sample_shape)
    avg = ts.mean()
    var = ts.var(ddof=1)
    np.testing.assert_allclose(avg, rate, atol=1e-2)
    np.testing.assert_allclose(var, rate, atol=1e-2)
