"""Stamox: Stats Models in JAX Library"""

import sys

from stamox import anova
from stamox import cluster
from stamox import distribution
from stamox import hypothesis
from stamox import math
from stamox import maps
from stamox import core
from stamox import basic
from stamox import regression
from stamox import decomposition
from stamox import sample


def _check_py_version():
    if sys.version_info[0] < 3:
        raise Exception("Please use Python 3. Python 2 is not supported.")


_check_py_version()


__all__ = [
    "anova",
    "cluster",
    "distribution",
    "hypothesis",
    "math",
    "maps",
    "core",
    "basic",
    "regression",
    "decomposition",
    "sample",
]
