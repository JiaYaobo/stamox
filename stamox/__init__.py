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
    hypothesis,
    math,
    regression,
    sample,
    transformation,
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
    "hypothesis",
    "math",
    "regression",
    "sample",
    "transformation",
]
