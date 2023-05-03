"""Stamox: Stats Models in JAX Library"""

import sys

from stamox import (
    anova,
    basic,
    cluster,
    core,
    correlation,
    distribution,
    experimental,
    formula,
    functions,
    hypothesis,
    math,
    pipe_functions,
    regression,
    sample,
    transformation,
)
from stamox.core import (
    better_partial,
    Functional,
    make_partial_pipe,
    make_pipe,
    Pipe,
    pipe_jit,
    pipe_pmap,
    pipe_vmap,
    Pipeable,
    StateFunc,
)


def _check_py_version():
    if sys.version_info[0] < 3:
        raise Exception("Please use Python 3. Python 2 is not supported.")


_check_py_version()


__all__ = [
    "anova",
    "basic",
    "cluster",
    "core",
    "correlation",
    "distribution",
    "experimental",
    "formula",
    "functions",
    "hypothesis",
    "math",
    "pipe_functions",
    "regression",
    "sample",
    "transformation",
    "Functional",
    "make_partial_pipe",
    "make_pipe",
    "partial_pipe_jit",
    "partial_pipe_pmap",
    "partial_pipe_vmap",
    "pipe_jit",
    "pipe_pmap",
    "pipe_vmap",
    "Pipe",
    "Pipeable",
    "StateFunc",
    "better_partial",
]
