from typing import Tuple, Sequence, Union, Any

import equinox as eqx
from equinox import Module



class Pipe(eqx.Module):
    funcs: Tuple[Module, ...]

    def __init__(self, funcs: Sequence[Module]) -> None:
        self.funcs = tuple(funcs)

    def __call__(self, x: Any, *args, **kwargs):
        for fn in self.funcs:
            x = fn(x, *args, **kwargs)
        return x

    def __getitem__(self, i: Union[int, slice]) -> Module:
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
        return Pipe([*self.funcs, _next])
