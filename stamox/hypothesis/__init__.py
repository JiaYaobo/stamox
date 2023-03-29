from stamox.hypothesis._bartlett import bartlett_test
from stamox.hypothesis._base import HypoTest
from stamox.hypothesis._durbin_watson import durbin_watson_test
from stamox.hypothesis._friedman import friedman_test
from stamox.hypothesis._shapiro_wilk import shapiro_wilk_test


__all__ = [
    "HypoTest",
    "bartlett_test",
    "durbin_watson_test",
    "friedman_test",
    "shapiro_wilk_test",
]