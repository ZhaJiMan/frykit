from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.7.0"

DATA_DIRPATH = Path(__file__).parent / "data"
SHP_DIRPATH = DATA_DIRPATH / "shp"

# 不知道为什么 setup.py 的 python_requires 没用
if sys.version_info.minor < 10:
    raise RuntimeError("要求 Python>=3.10")
