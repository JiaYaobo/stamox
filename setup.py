import pathlib

from setuptools import find_packages, setup


HERE = pathlib.Path(__file__).parent

VERSION = "0.1.5"
PACKAGE_NAME = "stamox"
AUTHOR = "Jia Yaobo"
URL = "https://github.com/jiayaobo/stamox"

LICENSE = "Apache 2.0"
DESCRIPTION = "Accelerate Your Statistical Analysis with JAX."
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    "jax>=0.4.6",
    "jaxtyping>=0.2.14",
    "typing_extensions>=4.5.0",
    "equinox>=0.10.1",
    "jaxopt>=0.6",
    "pandas>=1.5.3",
    "patsy>=0.5.3",
    "tensorflow-probability>=0.19.0",
    "scipy",
]
TESTS_REQUIRES = ["pytest", "scipy", "numpy", "sklearn", "statsmodels", "pandas"]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "tests": TESTS_REQUIRES,
        "complete": INSTALL_REQUIRES + TESTS_REQUIRES,
    },
    packages=find_packages(),
    python_requires=">=3.8",
)
