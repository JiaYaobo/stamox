from stamox.core import make_pipe
from stamox.hypothesis._bartlett import bartlett_test_fun, BartlettTest
from stamox.hypothesis._base import HypoTest
from stamox.hypothesis._durbin_watson import durbin_watson_test, DurbinWatsonTest
from stamox.hypothesis._friedman import friedman_test_fun, FriedmanTest
from stamox.hypothesis._shapiro_wilk import shapiro_wilk_test_fun, ShapiroWilkTest


bartlett_test = make_pipe(bartlett_test_fun, name="bartlett_test")
friedman_test = make_pipe(friedman_test_fun, name="friedman_test")
shapiro_wilk_test = make_pipe(shapiro_wilk_test_fun, name="shapiro_wilk_test")

__all__ = [
    "HypoTest",
    "bartlett_test",
    "BartlettTest",
    "durbin_watson_test",
    "DurbinWatsonTest",
    "friedman_test",
    "FriedmanTest",
    "shapiro_wilk_test",
    "ShapiroWilkTest",
]
