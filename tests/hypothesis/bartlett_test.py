from stamox.hypothesis import bartlett as stx_bartlett
import numpy as np

from scipy.stats import bartlett
from stamox.core import Pipeable


from jax import config

config.update('jax_enable_x64', True)

a = np.array([8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99], dtype=np.float64)
b = np.array([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05], dtype=np.float64)
c = np.array([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98], dtype=np.float64)

print(bartlett(a, b, c).pvalue)
print(stx_bartlett(a, b, c).df)