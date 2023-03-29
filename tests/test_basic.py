"""Test Basic Functions"""
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.basic import mean, median, scale, std, var
from stamox.core import Pipeable


class BasicTest(jtest.JaxTestCase):
    def test_mean(self):
        a = np.random.normal(size=(10000, 3))
        self.assertAllClose(mean(a, axis=1), np.mean(a, axis=1))

    def test_pipe_mean(self):
        a = np.random.normal(size=(10000, 3))
        p = Pipeable(a) >> mean(axis=1)
        self.assertAllClose(p(a), np.mean(a, axis=1))

    def test_median(self):
        a = np.random.normal(size=(10000, 3))
        self.assertAllClose(median(a, axis=1), np.median(a, axis=1))

    def test_pipe_median(self):
        a = np.random.normal(size=(10000, 3))
        p = Pipeable(a) >> median(axis=1)
        self.assertAllClose(p(a), np.median(a, axis=1))

    def test_std(self):
        a = np.random.normal(size=(10000, 3))
        self.assertAllClose(std(a, axis=1), np.std(a, axis=1))

    def test_pipe_std(self):
        a = np.random.normal(size=(10000, 3))
        p = Pipeable(a) >> std(axis=1)
        self.assertAllClose(p(a), np.std(a, axis=1))

    def test_var(self):
        a = np.random.normal(size=(10000, 3))
        self.assertAllClose(var(a, axis=1), np.var(a, axis=1))

    def test_pipe_var(self):
        a = np.random.normal(size=(10000, 3))
        p = Pipeable(a) >> var(axis=1)
        self.assertAllClose(p(a), np.var(a, axis=1))

    def test_scale(self):
        a = np.random.normal(size=(10000, 3)) * 2
        self.assertAllClose(
            scale(a), (a - np.mean(a, axis=0)) / np.std(a, axis=0, ddof=1)
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
