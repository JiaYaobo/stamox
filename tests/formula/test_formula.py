"""Test Formula Wrapper"""
import pandas as pd
from absl.testing import absltest
from jax._src import test_util as jtest

from stamox import Pipeable
from stamox.formula import get_design_matrices


class FormulaTest(jtest.JaxTestCase):
    def test_formula_pandas(self):
        df = pd.DataFrame(
            {
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "x2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )
        matrices = get_design_matrices(df, "y ~ x1 + x2 + x1:x2")
        self.assertEqual(matrices.y.shape, (10, 1))
        self.assertEqual(matrices.X.shape, (10, 4))

    def test_pipe_formula_pandas(self):
        df = pd.DataFrame(
            {
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "x2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )
        matrices = (
            Pipeable(df) >> get_design_matrices(formula="y ~ x1 + x2 + x1:x2")
        )()
        self.assertEqual(matrices.y.shape, (10, 1))
        self.assertEqual(matrices.X.shape, (10, 4))


if __name__ == "__main__":
    absltest.main(testLoader=jtest.JaxTestLoader())
