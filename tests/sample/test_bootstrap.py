"""Test for Bootstrap Sampler."""
import jax.random as jrandom
import numpy as np

import stamox.functions as F
import stamox.pipe_functions as PF
from stamox import Pipeable
from stamox.basic import mean


def test_bootstrap_sampler():
    key = jrandom.PRNGKey(20010813)
    X = jrandom.normal(key=key, shape=(1000, 3))
    S = F.bootstrap_sample(X, 5, key=key)
    np.testing.assert_equal(S.shape, (5, 1000, 3))


def test_partial_pipe_boostrap_sampler():
    key = jrandom.PRNGKey(20010813)
    X = jrandom.normal(key=key, shape=(1000, 3))
    h = Pipeable(X) >> PF.bootstrap_sample(num_samples=5, key=key)
    S = h()
    np.testing.assert_equal(S.shape, (5, 1000, 3))


def test_bootstrap():
    key = jrandom.PRNGKey(20010813)
    X = jrandom.normal(key=key, shape=(1000, 3))
    S = F.bootstrap(X, mean, 100, key=key)
    np.testing.assert_allclose(mean(S), mean(X), atol=1e-2)


def test_partial_pipe_bootstrap():
    key = jrandom.PRNGKey(20010813)
    X = jrandom.normal(key=key, shape=(1000, 3))
    h = Pipeable(X) >> PF.bootstrap(call=mean, num_samples=100, key=key)
    S = h()
    np.testing.assert_allclose(mean(S), mean(X), atol=1e-2)
