"""Test Basic Functions"""
import numpy as np

from stamox.basic import mean, median, scale, sd, var
from stamox.core import Pipeable


def test_mean():
    a = np.random.normal(size=(10000, 3))
    np.testing.assert_allclose(mean(a, axis=1), np.mean(a, axis=1))


def test_pipe_mean():
    a = np.random.normal(size=(10000, 3))
    p = Pipeable(a) >> mean(axis=1)
    np.testing.assert_allclose(p(a), np.mean(a, axis=1))


def test_median():
    a = np.random.normal(size=(10000, 3))
    np.testing.assert_allclose(median(a, axis=1), np.median(a, axis=1))


def test_pipe_median():
    a = np.random.normal(size=(10000, 3))
    p = Pipeable(a) >> median(axis=1)
    np.testing.assert_allclose(p(a), np.median(a, axis=1))


def test_std():
    a = np.random.normal(size=(10000, 3))
    np.testing.assert_allclose(sd(a, axis=1), np.std(a, axis=1))


def test_pipe_std():
    a = np.random.normal(size=(10000, 3))
    p = Pipeable(a) >> sd(axis=1)
    np.testing.assert_allclose(p(a), np.std(a, axis=1))


def test_var():
    a = np.random.normal(size=(10000, 3))
    np.testing.assert_allclose(var(a, axis=1), np.var(a, axis=1))


def test_pipe_var():
    a = np.random.normal(size=(10000, 3))
    p = Pipeable(a) >> var(axis=1)
    np.testing.assert_allclose(p(a), np.var(a, axis=1))


def test_scale():
    a = np.random.normal(size=(10000, 3)) * 2
    np.testing.assert_allclose(
        scale(a), (a - np.mean(a, axis=0)) / np.std(a, axis=0, ddof=1)
    )
