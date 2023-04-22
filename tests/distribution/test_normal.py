"""Test for normal distribution"""
import jax.random as jrand
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest
from scipy.stats import norm

import stamox.pipe_functions as PF
from stamox.distribution import dnorm, pnorm, qnorm, rnorm


np.random.seed(1)


def test_rnorm():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    mean = 0.5
    sd = 2.0
    norms = rnorm(key, mean=mean, sd=sd, sample_shape=sample_shape)
    avg = norms.mean()
    var = norms.var(ddof=1)
    np.testing.assert_allclose(avg, mean, atol=1e-2)
    np.testing.assert_allclose(var, sd**2, atol=1e-2)


def test_pnorm():
    x = np.random.normal(0, 1, 1000)
    mean = 0.0
    sd = 1.0
    p = pnorm(x, mean, sd, dtype=np.float64)
    true_p = norm(mean, sd).cdf(x)
    np.testing.assert_allclose(p, true_p)


def test_qnorm():
    q = np.random.uniform(0, 0.999, 1000)
    x = qnorm(q, dtype=np.float64)
    true_x = norm.ppf(q)
    np.testing.assert_allclose(x, true_x)


def test_dnorm():
    x = np.random.normal(0, 1, 1000)
    grads = dnorm(x, mean=0.0, sd=1.0, dtype=np.float64)
    true_grads = norm.pdf(x)
    np.testing.assert_allclose(grads, true_grads)


def test_partial_pnorm():
    x = np.array([-1.96, -1.645, -1.0, 0, 1.0, 1.645, 1.96])
    mean = 0.0
    sd = 1.0
    p = PF.pnorm(mean=mean, sd=sd)(x)
    true_p = np.array(
        [0.02499789, 0.04998492, 0.15865527, 0.5, 0.8413447, 0.95001507, 0.9750021]
    )
    np.testing.assert_allclose(p, true_p, atol=1e-6)


def test_partial_qnorm():
    q = np.array(
        [0.02499789, 0.04998492, 0.15865527, 0.5, 0.8413447, 0.95001507, 0.9750021]
    )
    x = PF.qnorm(mean=0.0, sd=1.0)(q)
    true_x = np.array([-1.96, -1.645, -1.0, 0, 1.0, 1.645, 1.96])
    np.testing.assert_allclose(x, true_x, atol=1e-6)


def test_partial_dnorm():
    x = np.array([-1.96, -1.645, -1.0, 0, 1.0, 1.645, 1.96])
    grads = PF.dnorm(mean=0.0, sd=1.0)(x)
    true_grads = np.array(
        [
            0.05844094,
            0.10311081,
            0.24197072,
            0.39894228,
            0.24197072,
            0.10311081,
            0.05844094,
        ]
    )
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_rnorm():
    key = jrand.PRNGKey(19751002)
    sample_shape = (1000000,)
    mean = 0.5
    sd = 2.0
    norms = PF.rnorm(mean=mean, sd=sd, sample_shape=sample_shape)(key)
    avg = norms.mean()
    var = norms.var(ddof=1)
    np.testing.assert_allclose(avg, mean, atol=1e-2)
    np.testing.assert_allclose(var, sd**2, atol=1e-2)
