from jaxtyping import ArrayLike, Int, Bool, Float
from equinox.nn import Linear

from ..core import StateFunc

class RegState(StateFunc):
    linear: Linear
    in_features: Int
    out_features: Int
    use_intercept: Bool
    df_model: Float
    df_residul: Float
    def __init__(self, in_features, out_features, use_intercept, *, key):
        super().__init__(name='RegState', fn=None)
        self.in_features = in_features
        self.out_features = out_features
        self.use_intercept = use_intercept
        self.linear = Linear(in_features=in_features, out_features=out_features, use_bias=use_intercept, key=key)
        self.df_model = 0.
        self.df_residul = 0.
    
    @property
    def params(self):
        if self.use_intercept is True:
            return self.linear.weight, self.linear.bias
        return self.linear.weight

    def __call__(self, X, *args, **kwargs):
        return self.linear(X)