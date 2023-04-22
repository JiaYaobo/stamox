"""Test Linear Regression"""
import numpy as np
import pandas as pd

import stamox.pipe_functions as PF
from stamox.core import Pipeable
from stamox.regression import lm


def test_lm_fullset():
    X = np.random.uniform(size=(1000, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
    data = pd.DataFrame(
        np.concatenate([X, y.reshape((-1, 1))], axis=1),
        columns=["x1", "x2", "x3", "y"],
    )

    res = lm(data, "y ~ x1 + x2 + x3")
    np.testing.assert_allclose(
        res.params.reshape(1, 4),
        np.array([[1.0, 3.0, 2.0, -7.0]]),
        atol=1e-5,
    )
    np.testing.assert_equal(res.df_resid, 996)
    np.testing.assert_equal(res.df_model, 3)


def test_lm_subset():
    subset = [i for i in range(500)]
    X = np.random.uniform(size=(1000, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
    data = pd.DataFrame(
        np.concatenate([X, y.reshape((-1, 1))], axis=1),
        columns=["x1", "x2", "x3", "y"],
    )

    res = lm(data, "y ~ x1 + x2 + x3", subset=subset)
    np.testing.assert_allclose(
        res.params.reshape(1, 4),
        np.array([[1.0, 3.0, 2.0, -7.0]]),
        atol=1e-5,
    )
    np.testing.assert_equal(res.df_resid, 496)
    np.testing.assert_equal(res.df_model, 3)


def test_lm_weighted():
    weights = np.abs(np.random.uniform(size=(1000, 1)))
    X = np.random.uniform(size=(1000, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
    data = pd.DataFrame(
        np.concatenate([X, y.reshape((-1, 1))], axis=1),
        columns=["x1", "x2", "x3", "y"],
    )

    res = lm(data, "y ~ x1 + x2 + x3", weights=weights)
    np.testing.assert_allclose(
        res.params.reshape(1, 4),
        np.array([[1.0, 3.0, 2.0, -7.0]]),
        atol=1e-5,
    )
    np.testing.assert_equal(res.df_resid, 996)
    np.testing.assert_equal(res.df_model, 3)


def test_lm_svd():
    X = np.random.uniform(size=(1000, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
    data = pd.DataFrame(
        np.concatenate([X, y.reshape((-1, 1))], axis=1),
        columns=["x1", "x2", "x3", "y"],
    )

    res = lm(data, "y ~ x1 + x2 + x3", method="svd")
    np.testing.assert_allclose(
        res.params.reshape(1, 4),
        np.array([[1.0, 3.0, 2.0, -7.0]]),
        atol=1e-5,
    )


def test_lm_inv():
    X = np.random.uniform(size=(1000, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
    data = pd.DataFrame(
        np.concatenate([X, y.reshape((-1, 1))], axis=1),
        columns=["x1", "x2", "x3", "y"],
    )

    res = lm(data, "y ~ x1 + x2 + x3", method="inv")
    np.testing.assert_allclose(
        res.params.reshape(1, 4),
        np.array([[1.0, 3.0, 2.0, -7.0]]),
        atol=1e-5,
    )


def test_lm_with_NA():
    X = np.random.uniform(size=(1000, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
    data = pd.DataFrame(
        np.concatenate([X, y.reshape((-1, 1))], axis=1),
        columns=["x1", "x2", "x3", "y"],
    )
    data["x1"][0] = np.nan
    data["x2"][1] = np.nan
    data["x3"][2] = np.nan
    data["y"][3] = np.nan

    res = lm(data, "y ~ x1 + x2 + x3")
    np.testing.assert_allclose(
        res.params.reshape(1, 4),
        np.array([[1.0, 3.0, 2.0, -7.0]]),
        atol=1e-5,
    )


def test_pipe_lm():
    X = np.random.uniform(size=(1000, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] - 7 * X[:, 2] + 1.0
    data = pd.DataFrame(
        np.concatenate([X, y.reshape((-1, 1))], axis=1),
        columns=["x1", "x2", "x3", "y"],
    )

    res = (Pipeable(data) >> PF.lm(formula="y ~ x1 + x2 + x3"))()
    np.testing.assert_allclose(
        res.params.reshape(1, 4),
        np.array([[1.0, 3.0, 2.0, -7.0]]),
        atol=1e-5,
    )


def test_lm_list_input():
    X = np.random.uniform(size=(1000, 3))
    X = np.concatenate([np.ones((1000, 1)), X], axis=1)
    y = 3 * X[:, 1] + 2 * X[:, 2] - 7 * X[:, 3] + 1.0
    data = [y, X]
    res = lm(data)
    np.testing.assert_allclose(
        res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
    )


def test_lm_tuple_input():
    X = np.random.uniform(size=(1000, 3))
    X = np.concatenate([np.ones((1000, 1)), X], axis=1)
    y = 3 * X[:, 1] + 2 * X[:, 2] - 7 * X[:, 3] + 1.0
    data = (y, X)
    res = lm(data)
    np.testing.assert_allclose(
        res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
    )


def test_lm_ndarray_input():
    X = np.random.uniform(size=(1000, 3))
    X = np.concatenate([np.ones((1000, 1)), X], axis=1)
    y = 3 * X[:, 1] + 2 * X[:, 2] - 7 * X[:, 3] + 1.0
    data = np.concatenate([y.reshape((-1, 1)), X], axis=1)
    res = lm(data)
    np.testing.assert_allclose(
        res.params.reshape(1, 4), np.array([[1.0, 3.0, 2.0, -7.0]]), atol=1e-5
    )
