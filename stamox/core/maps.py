from functools import partial, wraps
from typing import Any, Callable, Hashable, ParamSpec, TypeVar

from equinox import filter_pmap, filter_vmap

from .base import Functional


P = ParamSpec("P")
T = TypeVar("T")


def pipe_vmap(
    func: Callable[P, T],
    *,
    in_axes=0,
    out_axes=0,
    axis_name: Hashable = None,
    axis_size: int | None = None,
    name: str = "Anonymous"
) -> Callable[P, T]:
    """Make a Function Pipe Vmapped

    Args:
        func (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    if name is None and func is not None:
        name = func.__name__

    @wraps(func)
    def wrap(func: Callable[P, T]) -> Callable:
        if isinstance(func, Functional):
            func = func.func
        fn = filter_vmap(
            func,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
        )
        @wraps(func)
        def create_functional(*args):
            return Functional(name=name, fn=fn)(*args)
        return create_functional

    return wrap if func is None else wrap(func)


def partial_pipe_vmap(
    func: Callable[P, T], *, name: str = "Anonymous"
) -> Callable[P, T]:
    """Make a Partial Function Pipe Vmapped
    Args:
        func (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """
    if name is None and func is not None:
        if isinstance(func, Functional):
            name = func.name
        name = func.__name__

    @wraps(func)
    def wrap(func: Callable[P, T]) -> Callable:
        if isinstance(func, Functional):
            func = func.func

        @wraps(func)
        def partial_fn(
            x: Any = None,
            *args,
            in_axes=0,
            out_axes=0,
            axis_name: Hashable = None,
            axis_size: int | None = None,
            **kwargs
        ):
            fn = partial(func, **kwargs)
            fn = filter_vmap(
                fn,
                in_axes=in_axes,
                out_axes=out_axes,
                axis_name=axis_name,
                axis_size=axis_size,
            )
            if x is not None:
                return fn(x, *args)
            return Functional(name=name, fn=fn)

        return partial_fn

    return wrap if func is None else wrap(func)


def pipe_pmap(
    func: Callable[P, T],
    *,
    in_axes=0,
    out_axes=0,
    axis_name: Hashable = None,
    axis_size: int | None = None,
    name: str = "Anonymous"
) -> Callable[P, T]:
    """Make a Function Pipe Vmapped

    Args:
        func (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """

    if name is None and func is not None:
        name = func.__name__

    @wraps(func)
    def wrap(func: Callable[P, T]) -> Callable:
        if isinstance(func, Functional):
            func = func.func
        fn = filter_pmap(
            func,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
        )

        @wraps(func)
        def create_functional(*args):
            return Functional(name=name, fn=fn)(*args)
        return create_functional

    return wrap if func is None else wrap(func)


def partial_pipe_pmap(
    func: Callable[P, T], *, name: str = "Anonymous"
) -> Callable[P, T]:
    """Make a Partial Function Pipe Vmapped
    Args:
        func (Callable): Function or Callable Class
        params (PyTree, optional): Params For Function. Defaults to None.
        name (str, optional): Name of the Function. Defaults to "Anonymous".
    """
    if name is None and func is not None:
        name = func.__name__

    @wraps(func)
    def wrap(func: Callable[P, T]) -> Callable:
        if isinstance(func, Functional):
            func = func.func

        @wraps(func)
        def partial_fn(
            x: Any = None,
            *args,
            in_axes=0,
            out_axes=0,
            axis_name: Hashable = None,
            axis_size: int | None = None,
            **kwargs
        ):
            fn = partial(func, **kwargs)
            fn = filter_pmap(
                fn,
                in_axes=in_axes,
                out_axes=out_axes,
                axis_name=axis_name,
                axis_size=axis_size,
            )
            if x is not None:
                return fn(x, *args, **kwargs)
            return Functional(name=name, fn=fn, is_partial=True)

        return partial_fn

    return wrap if func is None else wrap(func)
