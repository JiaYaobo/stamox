import functools as ft

import jax
import jax.numpy as jnp
from jax.lax import linear_solve_p
from jax import jit, vamp

from ..base import ParamModel, NonParamModel, SemiParamModel
