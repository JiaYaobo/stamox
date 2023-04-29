"""Test for Durbin-Watson test"""
import numpy as np
from equinox import filter_jit
from statsmodels.stats.stattools import durbin_watson

import stamox.pipe_functions as PF
from stamox import Pipeable
from stamox.hypothesis import durbin_watson_test


def test_durbin_waston():
    x = np.array([0.1, 0.2, 0.3] * 50, dtype=np.float32)
    state = durbin_watson_test(x)
    np.testing.assert_allclose(state.statistic, np.array(durbin_watson(x)), atol=1e-3)


def test_pipe_durbin_waston():
    x = np.array([0.1, 0.2, 0.3] * 50, dtype=np.float32)
    h = Pipeable(x) >> PF.durbin_watson_test
    state = h()
    np.testing.assert_allclose(state.statistic, np.array(durbin_watson(x)), atol=1e-3)


def test_pipe_durbin_waston_jit():
    x = np.array([0.1, 0.2, 0.3] * 50, dtype=np.float32)
    h = filter_jit(Pipeable(x) >> PF.durbin_watson_test)
    state = h()
    np.testing.assert_allclose(state.statistic, np.array(durbin_watson(x)), atol=1e-3)
