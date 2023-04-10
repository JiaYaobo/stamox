"""Test for Special Functions"""
import scipy.special as scp_special
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox.math.special import fdtr, fdtrc, fdtri


class SpecialTest(jtest.JaxTestCase):
    def test_fdtri(self):
        df1 = 10
        df2 = 10
        p = 0.5
        x = fdtri(df1, df2, p)
        true_x = scp_special.fdtri(df1, df2, p)
        self.assertAllClose(x, true_x)

    def test_fdtr(self):
        df1 = 10
        df2 = 10
        x = 0.5
        p = fdtr(df1, df2, x)
        true_p = scp_special.fdtr(df1, df2, x)
        self.assertAllClose(p, true_p)

    def test_fdtrc(self):
        df1 = 10
        df2 = 10
        x = 0.5
        p = fdtrc(df1, df2, x)
        true_p = scp_special.fdtrc(df1, df2, x)
        self.assertAllClose(p, true_p)


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
