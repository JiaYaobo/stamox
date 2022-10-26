import jax
import equinox as eqx


class Model(eqx.Module):

    def __init__(self) -> None:
        super().__init__()

    def set_mode(self):
        pass

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)
