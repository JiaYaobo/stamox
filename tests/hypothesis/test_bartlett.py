"""Test for Bartlett test"""
import numpy as np
from equinox import filter_jit

import stamox.pipe_functions as PF
from stamox.core import Pipeable
from stamox.hypothesis import bartlett_test


def test_bartlett():
    a = np.array(
        [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99],
    )
    b = np.array(
        [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05],
    )
    c = np.array(
        [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98],
    )
    state = bartlett_test(a, b, c)
    np.testing.assert_allclose(state.p_value, np.array(1.1254782518834628e-05))


def test_pipe_bartlett():
    a = np.array(
        [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99],
    )
    b = np.array(
        [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05],
    )
    c = np.array(
        [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98],
    )
    h = Pipeable([a, b, c]) >> PF.bartlett_test
    state = h()
    np.testing.assert_allclose(state.p_value, np.array(1.1254782518834628e-05))


def test_pipe_bartlett_jit():
    a = np.array(
        [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99],
    )
    b = np.array(
        [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05],
    )
    c = np.array(
        [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98],
    )
    h = Pipeable([a, b, c]) >> filter_jit(PF.bartlett_test)
    state = h()
    np.testing.assert_allclose(state.p_value, np.array(1.1254782518834628e-05))
