import inspect
from typing import TypeVar


_T = TypeVar("_T")

Self = TypeVar("Self", bound="better_partial")


class better_partial:
    __slots__ = (
        "func",
        "keywords",
        "__dict__",
        "args_name",
        "use_argsonly",
        "__weakref__",
    )

    def __new__(cls, func, /, *args, **keywords) -> Self:
        if not callable(func):
            raise TypeError("the first argument must be callable")
        if "use_argsonly" in keywords:
            use_argsonly = keywords.pop("use_argsonly")
        else:
            use_argsonly = False
        if hasattr(func, "func"):
            keywords = {**func.keywords, **keywords}
            func = func.func
        self = super(better_partial, cls).__new__(cls)
        # get all args name
        if hasattr(func, "_static_fun"):
            args_name = inspect.getfullargspec(func._static_fun[0][0]).args
        elif hasattr(func, "_fun_value_and_grad"):
            args_name = inspect.getfullargspec(func._fun_value_and_grad._fun).args
        elif hasattr(func, "__dict__") and "__wrapped__" in func.__dict__:
            args_name = inspect.getfullargspec(func.__dict__["__wrapped__"]).args
        else:
            args_name = inspect.getfullargspec(func).args
        keywords_default = {}
        self.use_argsonly = use_argsonly
        self.args_name = args_name
        self.func = func
        unassign_argsname = [arg for arg in args_name if arg not in keywords]
        # get default argnames and value
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            if param.default != inspect.Parameter.empty:
                keywords_default[param.name] = param.default
        for i, arg in enumerate(unassign_argsname):
            if i < len(args):
                keywords[arg] = args[i]
        self.keywords = keywords_default
        self.keywords.update(keywords)

        return self

    def __call__(self, *args, **keywords) -> _T:
        keywords = {**self.keywords, **keywords}
        unassign_or_none_argsname = [
            arg
            for arg in self.args_name
            if arg not in keywords or keywords[arg] is None
        ]
        for i, arg in enumerate(unassign_or_none_argsname):
            if i < len(args):
                keywords[arg] = args[i]
        if self.use_argsonly:
            arg_vals = [keywords[arg] for arg in self.args_name]
            return self.func(*arg_vals)
        return self.func(**keywords)
