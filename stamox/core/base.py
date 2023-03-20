import equinox as eqx
from jaxtyping import PyTree
from .pipe import Pipe
    

class Functional(eqx.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)
    
    def __rshift__(self, _next):
        return Pipe([self, _next])


class StateFunc(Functional):
    _name: str
    _parmas: PyTree

    def __init__(self) -> None:
        super().__init__()
        self._name = None
        self._parmas = None

    @property
    def name(self):
        return self._name
    
    @property
    def params(self):
        return self._params
    
    def copy(self):
        return StateFunc(_name=self.name, _params=self.params)

    def __call__(cls, *args, **kwargs):
        pass


class StatelessFunc(Functional):
    _name: str

    def __init__(self) -> None:
        super().__init__()
        self._name = None
    
    def copy(self):
        return StateFunc(_name=self.name)

    def __call__(cls, *args, **kwargs):
        pass
