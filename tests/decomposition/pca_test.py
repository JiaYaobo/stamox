import numpy as np

from stamox.decomposition import PCA


X = np.random.randn(100, 5)

pca = PCA(n_components=2)

X_pca = pca(X)

print(X_pca.components)
