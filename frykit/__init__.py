import sys
from pathlib import Path

__version__ = '0.5.4'

DATA_DIRPATH = Path(__file__).parent / 'data'

# 不知道为什么setup.py的python_requires没用.
if sys.version_info.minor < 9:
    raise RuntimeError('要求Python>=3.9')
