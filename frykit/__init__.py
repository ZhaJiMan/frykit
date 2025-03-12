import sys

from frykit.option import *

__version__ = "0.7.0"

# 不知道为什么 setup.py 的 python_requires 没用
if sys.version_info.minor < 10:
    raise RuntimeError("要求 Python>=3.10")
