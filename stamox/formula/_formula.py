from typing import List, NamedTuple

import jax.numpy as jnp
import pandas as pd
import patsy
from jaxtyping import Array

from ..core import make_partial_pipe


class Matrices(NamedTuple):
    y: Array
    X: Array
    y_names: List[str]
    X_names: List[str]


@make_partial_pipe
def get_design_matrices(
    data: pd.DataFrame,
    formula: str,
    eval_env=0,
    NA_action="drop",
    return_type="matrix",
    dtype=jnp.float32,
) -> Matrices:
    """Get design matrices from a formula and data.

    Args:
        data: A pandas DataFrame containing the data.
        formula: A string representation of a formula.
        eval_env: An environment mapping column names to arrays which patsy can use when evaluating a formula.
        NA_action: A string specifying how to handle missing values. One of "drop", "raise", or "na_action".
        return_type: A string specifying the return type. One of "matrix", "dataframe", or "design_info".

    Returns:
        Matrices: A named tuple containing the design matrices for the response and predictors, as well as the names of the columns in each matrix.
    """
    y, X = patsy.dmatrices(
        formula, data, eval_env=eval_env, NA_action=NA_action, return_type=return_type
    )
    return Matrices(
        jnp.asarray(y, dtype=dtype),
        jnp.asarray(X, dtype=dtype),
        y.design_info.column_names,
        X.design_info.column_names,
    )
