import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..core import StateFunc


class RegState(StateFunc):
    in_features: int
    out_features: int
    _coefs: ArrayLike
    _df_resid: int
    _df_model: int
    _n_obs: int
    _resid: ArrayLike
    _fitted_values: ArrayLike
    _rank: int
    _dtype: jnp.dtype

    def __init__(
        self,
        in_features,
        out_features,
        coefs=None,
        df_resid=None,
        df_model=None,
        n_obs=None,
        resid=None,
        fitted_values=None,
        rank=None,
        dtype=jnp.float32,
        name="RegState",
    ):
        super().__init__(name=name, fn=None)
        self.in_features = in_features
        self.out_features = out_features
        if coefs is None:
            coefs = jnp.zeros((in_features, out_features))
        self._coefs = coefs
        self._df_resid = df_resid
        self._df_model = df_model
        self._n_obs = n_obs
        self._resid = resid
        self._fitted_values = fitted_values
        self._rank = rank
        self._dtype = dtype

    @property
    def params(self):
        return self._coefs

    @property
    def coefs(self):
        return self._coefs

    @property
    def coefs_X(self):
        return self.params[1:, :]

    @property
    def intercept(self):
        return self.params[0, :].reshape(-1, 1)

    @property
    def df_resid(self):
        return self._df_resid

    @property
    def df_model(self):
        return self._df_model

    @property
    def n_obs(self):
        return self._n_obs

    @property
    def resid(self):
        return self._resid

    @property
    def fitted_values(self):
        return self._fitted_values

    @property
    def rank(self):
        return self._rank

    @property
    def dtype(self):
        return self._dtype
    
    def _transform(self, X):
        if X.shape[1] == self.in_features:
            return jnp.matmul(X, self.params)
        else:
            return jnp.matmul(X, self.coefs_X) + self.intercept

    def _predict(self, X):
        return self._transform(X)

    def _summary(self):
        return "Not Implement Yet"


