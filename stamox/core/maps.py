from functools import wraps
from typing import Callable, Hashable, TypeVar

from equinox import filter_pmap, filter_vmap
from equinox.vmap_pmap import if_array

from .base import Functional


T = TypeVar("T")


def pipe_vmap(
    func: Callable[..., T] = None,
    *,
    in_axes=if_array(0),
    out_axes=if_array(0),
    axis_name: Hashable = None,
    axis_size: int = None,
    name: str = None
) -> Callable[..., T]:
    """Creates a functional from a function with vmap.

    Args:
        func: The function to be wrapped.
        in_axes: The number of input axes.
        out_axes: The number of output axes.
        axis_name: The name of the axis.
        axis_size: The size of the axis.
        name: The name of the functional. If not provided, the name of the
        function is used.

    Returns:
        A callable that creates a functional from the given function.

    Example:
        >>> from stamox import pipe_vmap
        >>> f = lambda x: x + 1
        >>> f = pipe_vmap(f)
        >>> g = f >> f >> f
        >>> g(jnp.array([1, 2, 3]))
        Array([4, 5, 6], dtype=int32)
    """
    if name is None and func is not None:
        if hasattr(func, "name"):
            name = func.name
        elif hasattr(func, "__name__"):
            name = func.__name__
        else:
            name = "none"

    @wraps(func)
    def wrap(func: Callable[..., T]) -> Callable:
        if isinstance(func, Functional):
            if func.func is not None:
                func = func.func
        fn = filter_vmap(
            func,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
        )
        return Functional(name=name, fn=fn, pipe_type='vmap')

    return wrap if func is None else wrap(func)


def pipe_pmap(
    func: Callable[..., T] = None,
    *,
    in_axes=0,
    out_axes=0,
    axis_name: Hashable = None,
    axis_size: int = None,
    name: str = None
) -> Callable[..., T]:
    """Creates a functional object from a given function.

    Args:
        func (Callable[P, T]): The function to be wrapped.
        in_axes (int): The number of input axes for the function.
        out_axes (int): The number of output axes for the function.
        axis_name (Hashable): The name of the axis.
        axis_size (int ): The size of the axis.
        name (str): The name of the functional object.

    Returns:
        Callable[P, T]: A callable object that wraps the given function.

    Example:
        >>> from stamox import pipe_pmap
        >>> f = lambda x: x + 1
        >>> f = pipe_pmap(f)
        >>> g = f >> f >> f
        >>> g(jnp.array([1, 2, 3]))
        Array([4, 5, 6], dtype=int32)
    """
    if name is None and func is not None:
        if hasattr(func, "name"):
            name = func.name
        elif hasattr(func, "__name__"):
            name = func.__name__
        else:
            name = "none"

    @wraps(func)
    def wrap(func: Callable[..., T]) -> Callable:
        if isinstance(func, Functional):
            if func.func is not None:
                func = func.func
        fn = filter_pmap(
            func,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
        )

        return Functional(name=name, fn=fn, pipe_type="pmap")

    return wrap if func is None else wrap(func)

