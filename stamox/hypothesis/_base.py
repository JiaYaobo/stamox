from typing import AnyStr

from jaxtyping import ArrayLike

from ..core import StateFunc


class HypoTest(StateFunc):
    statistic: ArrayLike
    parameters: ArrayLike
    p_value: ArrayLike
    estimate: ArrayLike
    null_value: ArrayLike
    alternative: AnyStr

    def __init__(
        self,
        statistic=None,
        parameters=None,
        p_value=None,
        estimate=None,
        null_value=None,
        alternative=None,
        name='HTest'
    ):
        super().__init__(name=name, fn=None)
        self.statistic = statistic
        self.parameters = parameters
        self.p_value = p_value
        self.estimate = estimate
        self.null_value = null_value
        self.alternative = alternative
    
