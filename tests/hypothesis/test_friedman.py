"""Test for Friedman test"""
import numpy as np
from equinox import filter_jit
from scipy.stats import friedmanchisquare

import stamox.pipe_functions as PF
from stamox import Pipeable
from stamox.hypothesis import friedman_test


def test_friedman():
    x3 = [
        np.array([7.0, 9.9, 8.5, 5.1, 10.3]),
        np.array([5.3, 5.7, 4.7, 3.5, 7.7]),
        np.array([4.9, 7.6, 5.5, 2.8, 8.4]),
        np.array([8.8, 8.9, 8.1, 3.3, 9.1]),
    ]

    state = friedman_test(x3[0], x3[1], x3[2], x3[3])
    osp_state = friedmanchisquare(x3[0], x3[1], x3[2], x3[3])
    np.testing.assert_allclose(
        state.statistic,
        np.array(osp_state.statistic, dtype=np.float32),
    )
    np.testing.assert_allclose(
        state.p_value,
        np.array(osp_state.pvalue, dtype=np.float32),
    )

def test_pipe_friedman():
    x3 = [
        np.array([7.0, 9.9, 8.5, 5.1, 10.3]),
        np.array([5.3, 5.7, 4.7, 3.5, 7.7]),
        np.array([4.9, 7.6, 5.5, 2.8, 8.4]),
        np.array([8.8, 8.9, 8.1, 3.3, 9.1]),
    ]
    dt = [x3[0], x3[1], x3[2], x3[3]]
    state = (Pipeable(dt) >> PF.friedman_test)()
    osp_state = friedmanchisquare(x3[0], x3[1], x3[2], x3[3])
    np.testing.assert_allclose(
        state.statistic,
        np.array(osp_state.statistic, dtype=np.float32),
    )
    np.testing.assert_allclose(
        state.p_value,
        np.array(osp_state.pvalue, dtype=np.float32),
    )

def test_pipe_friedman_jit():
    x3 = [
        np.array([7.0, 9.9, 8.5, 5.1, 10.3]),
        np.array([5.3, 5.7, 4.7, 3.5, 7.7]),
        np.array([4.9, 7.6, 5.5, 2.8, 8.4]),
        np.array([8.8, 8.9, 8.1, 3.3, 9.1]),
    ]
    dt = [x3[0], x3[1], x3[2], x3[3]]
    state = filter_jit(Pipeable(dt) >> PF.friedman_test)()
    osp_state = friedmanchisquare(x3[0], x3[1], x3[2], x3[3])
    np.testing.assert_allclose(
        state.statistic.astype(np.float32),
        np.array(osp_state.statistic, dtype=np.float32),
    )
    np.testing.assert_allclose(
        state.p_value,
        np.array(osp_state.pvalue, dtype=np.float32),
    )
