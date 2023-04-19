from stamox.hypothesis._bartlett import bartlett_test, BartlettTest
from stamox.hypothesis._base import HypoTest
from stamox.hypothesis._durbin_watson import durbin_watson_test, DurbinWatsonTest
from stamox.hypothesis._friedman import friedman_test, FriedmanTest
from stamox.hypothesis._shapiro_wilk import shapiro_wilk_test, ShapiroWilkTest


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
