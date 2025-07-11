from __future__ import annotations

from functools import cache
from pathlib import Path

from frykit.configuration import config as config
from frykit.option import *

__version__ = "0.7.5"


@cache
def get_data_dirpath() -> Path:
    """获取 frykit_data 的数据目录"""
    try:
        from frykit_data import DATA_DIRPATH

        return DATA_DIRPATH
    except ImportError:
        raise ImportError("需要地图数据请 pip install frykit[data]")
