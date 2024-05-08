import shutil
import warnings
from collections.abc import Iterator, Sequence
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

from frykit._typing import PathType


def new_dir(dirpath: PathType) -> None:
    '''新建目录'''
    dirpath = Path(dirpath)
    if not dirpath.exists():
        dirpath.mkdir(parents=True)


def del_dir(dirpath: PathType) -> None:
    '''删除目录。目录不存在时会报错。'''
    dirpath = Path(dirpath)
    shutil.rmtree(str(dirpath))


def renew_dir(dirpath: PathType) -> None:
    '''重建目录'''
    dirpath = Path(dirpath)
    if dirpath.exists():
        shutil.rmtree(str(dirpath))
    dirpath.mkdir(parents=True)


def split_list(lst: Sequence, n: int) -> Iterator[Sequence]:
    '''将列表尽量等分为 n 份'''
    size, rest = divmod(len(lst), n)
    start = 0
    for i in range(n):
        step = size + 1 if i < rest else size
        stop = start + step
        yield lst[start:stop]
        start = stop


class DeprecationError(Exception):
    pass


def deprecator(
    new_func: Optional[Callable], raise_error: bool = False
) -> Callable:
    '''提示弃用的装饰器'''

    def decorator(old_func: Callable) -> Callable:
        info = f'{old_func.__name__} is deprecated'
        if new_func is not None:
            info += f', use {new_func.__name__} instead'

        @wraps(old_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if raise_error:
                raise DeprecationError(info)
            warnings.warn(info, DeprecationWarning, stacklevel=2)
            result = old_func(*args, **kwargs)
            return result

        return wrapper

    return decorator
