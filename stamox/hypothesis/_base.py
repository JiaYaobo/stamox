from jaxtyping import ArrayLike

from ..core import StateFunc


class HypoTest(StateFunc):
    _statistic: ArrayLike
    _parameters: ArrayLike
    _p_value: ArrayLike
    _estimate: ArrayLike
    _null_value: ArrayLike
    _alternative: str

    def __init__(
        self,
        statistic=None,
        parameters=None,
        p_value=None,
        estimate=None,
        null_value=None,
        alternative=None,
        name="HTest",
    ):
        super().__init__(name=name, fn=None)
        self._statistic = statistic
        self._parameters = parameters
        self._p_value = p_value
        self._estimate = estimate
        self._null_value = null_value
        self._alternative = alternative

    @property
    def statistic(self):
        return self._statistic

    @property
    def parameters(self):
        return self._parameters

    @property
    def p_value(self):
        return self._p_value
