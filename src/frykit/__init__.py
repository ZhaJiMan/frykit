from __future__ import annotations

from functools import cache
from pathlib import Path

from frykit.conf import config as config
from frykit.option import *
from frykit.utils import deprecator

__version__ = "0.7.5.post1"


@cache
def get_data_dir() -> Path:
    """获取 frykit_data 的数据目录"""
    try:
        # TODO: 在 frykit_data 里改名为 DATA_DIR
        from frykit_data import DATA_DIRPATH

        return DATA_DIRPATH
    except ImportError:
        raise ImportError("需要地图数据请 pip install frykit[data]")


@deprecator(alternative=get_data_dir)
def get_data_dirpath() -> Path:
    return get_data_dir()
