from functools import partial
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
        return Functional(name="vmapped_" + name, fn=fn)

    if func is None:
        return wrap

    return wrap(func)


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

    def wrap(func: Callable[P, T]) -> Callable:
        if isinstance(func, Functional):
            func = func.func

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

        return Functional(
            name="partial_vmapped_" + name, fn=partial_fn, is_partial=True
        )

    if func is None:
        return wrap

    return wrap(func)


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
        return Functional(name="pmapped_" + name, fn=fn)

    if func is None:
        return wrap

    return wrap(func)


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

    def wrap(func: Callable[P, T]) -> Callable:
        if isinstance(func, Functional):
            func = func.func

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

        return Functional(name="partial_pmapped_" + name, fn=partial_fn)

    if func is None:
        return wrap

    return wrap(func)
