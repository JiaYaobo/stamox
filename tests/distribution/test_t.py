"""Test for student t distribution"""
import jax.random as jrand
import numpy as np
from scipy.stats import t

import stamox.pipe_functions as PF
from stamox.distribution import dt, pt, qt, rt


np.random.seed(19751002)
key = jrand.PRNGKey(19751002)
sample_shape = (1000000,)


def test_rt():
    df = 6
    ts = rt(key, sample_shape, df, 0.0, 1.0)
    avg = ts.mean()
    var = ts.var(ddof=1)
    np.testing.assert_allclose(avg, 0.0, atol=1e-2)
    np.testing.assert_allclose(var, float(df / (df - 2)), atol=1e-2)


def test_pt():
    x = np.random.standard_t(3, 10000)
    p = pt(x, 3, dtype=np.float64)
    true_p = t.cdf(x, 3)
    np.testing.assert_allclose(p, true_p)


def test_qt():
    q = np.random.uniform(0, 0.999, 10000)
    x = qt(q, 3, dtype=np.float64)
    true_x = t.ppf(q, 3)
    np.testing.assert_allclose(x, true_x, atol=1e-7)


def test_dt():
    x = np.random.standard_t(6, 10)
    grads = dt(x, 6, dtype=np.float64)
    true_grads = t.pdf(x, 6)
    np.testing.assert_allclose(grads, true_grads)


def test_partial_pt():
    x = np.array([1.0, 1.645, 1.96, 2.65, 3.74])
    p = PF.pt(df=6)(x)
    true_p = np.array([0.8220412, 0.9244638, 0.9511524, 0.9809857, 0.9951888])
    np.testing.assert_allclose(p, true_p)


def test_partial_qt():
    q = np.array([0.8220412, 0.9244638, 0.9511524, 0.9809857, 0.9951888])
    x = PF.qt(df=6)(q)
    true_x = np.array([1.0, 1.645, 1.96, 2.65, 3.74])
    np.testing.assert_allclose(x, true_x, atol=1e-5)


def test_partial_dt():
    x = np.array([1.0, 1.645, 1.96, 2.65, 3.74])
    grads = PF.dt(df=6)(x)
    true_grads = np.array(
        [0.223142291, 0.104005269, 0.067716584, 0.025409420, 0.005672347]
    )
    np.testing.assert_allclose(grads, true_grads, atol=1e-6)


def test_partial_rt():
    rvs = PF.rt(df=6, loc=0.0, scale=1.0, sample_shape=sample_shape)(key)
    avg = rvs.mean()
    var = rvs.var(ddof=1)
    np.testing.assert_allclose(avg, 0.0, atol=1e-2)
    np.testing.assert_allclose(var, float(6 / (6 - 2)), atol=1e-2)
