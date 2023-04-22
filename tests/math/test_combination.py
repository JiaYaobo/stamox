"""Test for Combination"""
import numpy as np

from stamox.math.combination import combination


def test_comb():
    n = 5
    k = np.array([0, 1, 2, 3, 4, 5, 6, -1])
    combs = combination(k, n)
    true_combs = np.array([1, 5, 10, 10, 5, 1, 0, 0])
    np.testing.assert_array_equal(combs, true_combs)


def test_pick():
    n = 5
    k = np.array([0, 1, 2, 3, 4, 5, 6, -1])
    combs = combination(n=n)(k)
    true_combs = np.array([1, 5, 10, 10, 5, 1, 0, 0])
    np.testing.assert_array_equal(combs, true_combs)
