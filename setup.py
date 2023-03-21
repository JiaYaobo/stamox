import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'stamox'
AUTHOR = 'Jia Yaobo'
URL = 'https://github.com/jiayaobo/stamox'

LICENSE = 'Apache 2.0'
DESCRIPTION = 'High Performance Statistics Library'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = ['numpy>=1.23', 'pandas>=1.5.0', 'jax>=0.3.15', 'equinox>=0.8.0',
                    'chex>=0.1.5', 'tensorflow-probability>=0.18.0']
TESTS_REQUIRES = ['pytest']

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
        'tests': TESTS_REQUIRES,
        'complete': INSTALL_REQUIRES + TESTS_REQUIRES,
    },
    packages=find_packages(),
    python_requires='>=3'
)
