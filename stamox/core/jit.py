from functools import partial
from typing import Callable, Any

from equinox import filter_jit
from jaxtyping import PyTree

from .base import StateFunc, StatelessFunc


def pipe_jit(
    cls: Callable,
    params: PyTree = None,
    name: str = "Anonymous",
) -> Callable:
    """Make a Function Pipe Jitted

    Args:
        cls (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    def wrap(cls):
        fn = filter_jit(cls)
        if params is None:
            return StatelessFunc(name=name, fn=fn)
        else:
            return StateFunc(params=params, name=name, fn=fn)

    if cls is None:
        return wrap

    return wrap(cls)


def partial_pipe_jit(
    cls: Callable, params: PyTree = None, name: str = "Anonymous"
) -> Callable:
    """Make a Partial Function Pipe Jitted
    Args:
        cls (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    def wrap(cls) -> Callable:
        def partial_fn(x: Any=None, **kwargs):
            fn = filter_jit(cls)
            fn = partial(fn, **kwargs)
            if x is not None:
                return fn(x)
            if params is None:
                return StatelessFunc(name=name, fn=fn)
            else:
                return StateFunc(params=params, name=name, fn=fn)

        return partial_fn

    if cls is None:
        return wrap

    return wrap(cls)