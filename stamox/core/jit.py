from functools import partial
from typing import Any, Callable, ParamSpec, TypeVar

from equinox import filter_jit

from .base import Functional


P = ParamSpec("P")
T = TypeVar("T")


def pipe_jit(func: Callable[P, T], name: str = "Anonymous") -> Callable[P, T]:
    """Make a Function Pipe Jitted

    Args:
        func (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    if name is None and func is not None:
        name = func.__name__

    def wrap(func: Callable[P, T]) -> Callable:
        fn = filter_jit(func)
        return Functional(name="jitted_" + name, fn=fn)

    if func is None:
        return wrap

    return wrap(func)


def partial_pipe_jit(func: Callable[P, T], name: str = "Anonymous") -> Callable[P, T]:
    """Make a Partial Function Pipe Jitted
    Args:
        func (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """
    if name is None and func is not None:
        name = func.__name__

    def wrap(func: Callable[P, T]) -> Callable:
        def partial_fn(x: Any = None, *args, **kwargs):
            fn = filter_jit(func)
            fn = partial(fn, **kwargs)
            if x is not None:
                return fn(x, *args, **kwargs)
            return Functional(name=name, fn=fn)

        return Functional(name="partial_jitted_" + name, fn=partial_fn)

    if func is None:
        return wrap

    return wrap(func)
