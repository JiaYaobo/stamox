from typing import Callable, Optional, Any

import equinox as eqx


class Functional(eqx.Module):
    """General Function"""

    _name: str
    _fn: Callable

    def __init__(self, name: str = "Func", fn: Optional[Callable] = None):
        """Make a General Function

        Args:
            fn (Optional[Callable|None], optional): Callable object.
        """
        super().__init__()
        self._name = name
        self._fn = fn

    @property
    def name(self):
        return self._name

    def desc(self):
        """Description for the function"""
        pass

    def __call__(self, x: Any, *args, **kwargs):
        if self._fn is None:
            raise ValueError("No Callable Function to Call")
        return self._fn(x, *args, **kwargs)

    def __rshift__(self, _next):
        """Make Pipe"""
        from .pipe import Pipe

        return Pipe([self, _next])


class StateFunc(Functional):
    def __init__(self, name: str = "State", fn: Optional[Callable] = None):
        super().__init__(name, fn)

    def __repr__(self):
        return super().__repr__()

    def _tree_flatten(self):
        return super()._tree_flatten()

    def _summary(self):
        pass

    def __call__(self, *args, **kwargs):
        return self


