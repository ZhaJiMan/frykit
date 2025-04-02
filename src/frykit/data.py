from pathlib import Path

try:
    from frykit_data import DATA_DIRPATH  # type: ignore

    HAS_FRYKIT_DATA = True
except ImportError:
    HAS_FRYKIT_DATA = False


def get_data_dirpath() -> Path:
    """获取 frykit_data 的数据目录"""
    if not HAS_FRYKIT_DATA:
        raise ImportError("需要地图数据请 pip install frykit[data]")

    return DATA_DIRPATH  # type: ignore
