from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from frykit.conf import config

__version__ = "0.8.1"

__all__ = ["config", "get_data_dir"]


def get_data_dir() -> Path:
    """获取 frykit_data 的数据目录"""
    try:
        dirpath = files("frykit_data") / "data"
    except ModuleNotFoundError:
        raise ModuleNotFoundError("需要安装 frykit_data 包") from None

    # 字体文件需要存在于磁盘上
    if not isinstance(dirpath, Path):
        raise RuntimeError("frykit_data 包需要以普通文件夹形式安装")

    return dirpath
