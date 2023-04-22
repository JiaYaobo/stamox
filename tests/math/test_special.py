"""Test for Special Functions"""
import numpy as np
import scipy.special as scp_special

from stamox.math.special import fdtr, fdtrc, fdtri


def test_fdtri():
    df1 = 10
    df2 = 10
    p = 0.5
    x = fdtri(df1, df2, p)
    true_x = scp_special.fdtri(df1, df2, p)
    np.testing.assert_allclose(x, true_x)


def test_fdtr():
    df1 = 10
    df2 = 10
    x = 0.5
    p = fdtr(df1, df2, x)
    true_p = scp_special.fdtr(df1, df2, x)
    np.testing.assert_allclose(p, true_p)


def test_fdtrc():
    df1 = 10
    df2 = 10
    x = 0.5
    p = fdtrc(df1, df2, x)
    true_p = scp_special.fdtrc(df1, df2, x)
    np.testing.assert_allclose(p, true_p)
