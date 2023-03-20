import equinox as eqx

class Pipe(eqx.nn.Sequential):
    def __init__(self, funcs) -> None:
        super().__init__(funcs)

    def __rshift__(self, _next):
        return Pipe([self, _next])    


