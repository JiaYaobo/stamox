"""Test for Bootstrap Sampler."""
import jax.random as jrandom
import numpy as np

import stamox.pipe_functions as PF
from stamox.basic import mean
from stamox.sample import jackknife, jackknife_sample


def test_jackknife_sampler():
    key = jrandom.PRNGKey(20010813)
    X = jrandom.normal(key=key, shape=(1000, 3))
    S = jackknife_sample(X)
    np.testing.assert_equal(S.shape, (1000, 999, 3))


def test_partial_pipe_jackknife_sampler():
    key = jrandom.PRNGKey(20010813)
    X = jrandom.normal(key=key, shape=(1000, 3))
    h = PF.Pipeable(X) >> PF.jackknife_sample
    S = h()
    np.testing.assert_equal(S.shape, (1000, 999, 3))


def test_jackknife():
    key = jrandom.PRNGKey(20010813)
    X = jrandom.normal(key=key, shape=(1000, 3))
    S = jackknife(X, mean)
    np.testing.assert_allclose(mean(S), mean(X), atol=1e-2)


def test_partial_pipe():
    key = jrandom.PRNGKey(20010813)
    X = jrandom.normal(key=key, shape=(1000, 3))
    h =PF.Pipeable(X) >> PF.jackknife(call=mean)
    S = h()
    np.testing.assert_allclose(mean(S), mean(X), atol=1e-2)
