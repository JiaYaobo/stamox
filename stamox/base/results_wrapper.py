import equinox as eqx

class _BaseResult(eqx.Module):
    def __init__(self) -> None:
        super().__init__()
    

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)