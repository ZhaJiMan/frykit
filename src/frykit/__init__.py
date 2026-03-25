from __future__ import annotations

from pathlib import Path

from frykit.conf import config

__version__ = "0.8.1"

__all__ = ["config", "get_data_dir"]


def get_data_dir() -> Path:
    """获取 frykit_data 的数据目录"""
    try:
        import frykit_data
    except ModuleNotFoundError:
        raise ModuleNotFoundError("需要安装 frykit_data 包") from None
    else:
        # 字体文件需要存在于磁盘上
        return Path(frykit_data.__file__).parent / "data"
