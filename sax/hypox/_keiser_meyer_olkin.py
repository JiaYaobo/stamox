"""
The Kaiser–Meyer–Olkin (KMO) test is a statistical measure to determine how suited data is for factor analysis. 
The test measures sampling adequacy for each variable in the model and the complete model. 
The statistic is a measure of the proportion of variance among variables that might be common variance. 
The higher the proportion, the higher the KMO-value, the more suited the data is to factor analysis
"""
import functools as ft

import jax.numpy as jnp
from jax import jit, vmap