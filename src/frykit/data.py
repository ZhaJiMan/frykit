from pathlib import Path

try:
    from frykit_data import DATA_DIRPATH

    HAS_FRYKIT_DATA = True
except ImportError:
    HAS_FRYKIT_DATA = False


def get_data_dirpath() -> Path:
    """获取 frykit_data 的数据目录"""
    if not HAS_FRYKIT_DATA:
        raise ImportError("frykit_data 未安装")

    return DATA_DIRPATH  # type: ignore
