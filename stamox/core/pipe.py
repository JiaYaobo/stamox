from typing import Tuple, Sequence, Union, Any, Callable
from functools import partial

import equinox as eqx
from equinox import Module

from .base import Functional



class Pipe(eqx.Module):
    funcs: Tuple[Module, ...]

    def __init__(self, funcs: Sequence[Module]) -> None:
        self.funcs = tuple(funcs)

    def __call__(self, x: Any, *args, **kwargs):
        for fn in self.funcs:
            if not (isinstance(fn, Functional)):
                fn = fn()
            x = fn(x, *args, **kwargs)
        return x

    def __getitem__(self, i: Union[int, slice, str]) -> Module:
        if isinstance(i, int):
            return self.funcs[i]
        elif isinstance(i, slice):
            return Pipe(self.funcs[i])
        elif isinstance(i, str):
            _f = []
            i = i.lower()
            for f in self.funcs:
                if f.name.lower() == i:
                    _f.append(f)

            if len(_f) == 0:
                raise ValueError(f"No Function Names {i}")
            if len(_f) > 1:
                return Pipe(_f)
            else:
                return _f[0]
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported")

    def __iter__(self):
        yield from self.funcs

    def __len__(self):
        return len(self.funcs)

    def __rshift__(self, _next):
        if not isinstance(_next, Functional):
            _next = Functional(fn=_next)
        return Pipe([*self.funcs, _next])


class Pipeable(Functional):
    def __init__(self, value):
        self.value = value
    
    def __rshift__(self, *args, **kwargs):
        return self.value

def make_pipe(cls: Callable, name: str = "PipeableFunc", **kwargs) -> Callable:
    """Make a Function Pipable

    Args:
        cls (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    def wrap(cls):
        return Functional(name=name, fn=cls)

    if cls is None:
        return wrap

    return wrap(cls)


def make_partial_pipe(cls: Callable, name: str = "PipeableFunc", **kwargs) -> Callable:
    """Make a Partial Function Pipe
    Args:
        cls (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    def wrap(cls) -> Callable:
        def partial_fn(x: Any = None, **kwargs):
            fn = partial(cls, **kwargs)
            if x is not None:
                return fn(x)
            return Functional(name=name, fn=fn)

        return partial_fn

    if cls is None:
        return wrap

    return wrap(cls)