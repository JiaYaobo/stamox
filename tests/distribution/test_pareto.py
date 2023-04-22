"""Test for pareto distribution"""
import jax.random as jrand
import numpy as np

import stamox.pipe_functions as PF
from stamox.distribution import dpareto, ppareto, qpareto, rpareto


def test_rpareto():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    scale = 0.1
    alpha = 3.0
    ts = rpareto(key, sample_shape, scale, alpha)
    avg = ts.mean()
    var = ts.var(ddof=1)
    np.testing.assert_allclose(avg, alpha * scale / (alpha - 1), atol=1e-2)
    np.testing.assert_allclose(
        var, scale**2 * alpha / (((alpha - 1) ** 2) * (alpha - 2)), atol=1e-2
    )


def test_ppareto():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    scale = 0.1
    alpha = 2.0
    p = ppareto(x, scale, alpha)
    true_p = np.array([0.0000000, 0.7500000, 0.8888889, 0.9375000, 0.9600000])
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_qpareto():
    q = np.array([0.0000000, 0.7500000, 0.8888889, 0.9375000, 0.9600000])
    x = qpareto(q, 0.1, 2.0)
    true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(x, true_x, atol=1e-6)


def test_dpareto():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    grads = dpareto(x, 0.1, 2.0)
    true_grads = np.array([20.0000000, 2.5000000, 0.7407407, 0.3125000, 0.1600000])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_ppareto():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    scale = 0.1
    alpha = 2.0
    p = PF.ppareto(scale=scale, alpha=alpha)(x)
    true_p = np.array([0.0000000, 0.7500000, 0.8888889, 0.9375000, 0.9600000])
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_partial_qpareto():
    q = np.array([0.0000000, 0.7500000, 0.8888889, 0.9375000, 0.9600000])
    x = PF.qpareto(scale=0.1, alpha=2.0)(q)
    true_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(x, true_x, atol=1e-6)


def test_partial_dpareto():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    grads = PF.dpareto(scale=0.1, alpha=2.0)(x)
    true_grads = np.array([20.0000000, 2.5000000, 0.7407407, 0.3125000, 0.1600000])
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_rpareto():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    scale = 0.1
    alpha = 3.0
    ts = PF.rpareto(scale=scale, alpha=alpha, sample_shape=sample_shape)(key)
    avg = ts.mean()
    var = ts.var(ddof=1)
    np.testing.assert_allclose(avg, alpha * scale / (alpha - 1), atol=1e-2)
    np.testing.assert_allclose(
        var, scale**2 * alpha / (((alpha - 1) ** 2) * (alpha - 2)), atol=1e-2
    )
