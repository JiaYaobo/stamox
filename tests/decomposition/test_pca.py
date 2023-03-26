import numpy as np

from stamox.decomposition import PCA
from stamox.core import summary


X = np.random.randn(100, 5)

pca = PCA(n_components=2)

print(pca >> summary >> np.mean)

