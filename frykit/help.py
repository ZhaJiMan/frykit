import shutil
import warnings
from pathlib import PurePath, Path
from functools import wraps
from collections.abc import Sequence, Iterator
from typing import Any, Union, Optional, Callable

def new_dir(dirpath: Union[str, Path]) -> None:
    '''新建目录.'''
    dirpath = Path(dirpath)
    if not dirpath.exists():
        dirpath.mkdir(parents=True)

def del_dir(dirpath: Union[str, PurePath]) -> None:
    '''删除目录. 目录不存在时会报错.'''
    dirpath = Path(dirpath)
    shutil.rmtree(str(dirpath))

def renew_dir(dirpath: Union[str, Path]) -> None:
    '''重建目录.'''
    dirpath = Path(dirpath)
    if dirpath.exists():
        shutil.rmtree(str(dirpath))
    dirpath.mkdir(parents=True)

def split_list(lst: Sequence, n: int) -> Iterator[Sequence]:
    '''将列表尽量等分为n份.'''
    size, rest = divmod(len(lst), n)
    start = 0
    for i in range(n):
        step = size + 1 if i < rest else size
        stop = start + step
        yield lst[start:stop]
        start = stop

def deprecator(new_func: Optional[Callable]) -> Callable:
    '''提示弃用的装饰器.'''
    def decorator(old_func: Callable) -> Callable:
        info = f'{old_func.__name__} is deprecated'
        if new_func is not None:
            info += f', use {new_func.__name__} instead'
        @wraps(old_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(info, DeprecationWarning, stacklevel=2)
            result = old_func(*args, **kwargs)
            return result
        return wrapper
    return decorator