import shutil
import warnings
from collections.abc import Iterable, Iterator
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Optional, Union

from frykit._typing import PathType


def new_dir(dirpath: PathType) -> Path:
    '''新建目录'''
    dirpath = Path(dirpath)
    if not dirpath.exists():
        dirpath.mkdir(parents=True)

    return dirpath


def del_dir(dirpath: PathType) -> Path:
    '''删除目录。目录不存在时会报错。'''
    dirpath = Path(dirpath)
    shutil.rmtree(str(dirpath))

    return dirpath


def renew_dir(dirpath: PathType) -> Path:
    '''重建目录'''
    dirpath = Path(dirpath)
    if dirpath.exists():
        shutil.rmtree(str(dirpath))
    dirpath.mkdir(parents=True)

    return dirpath


def split_list(lst: list, n: int) -> Iterator[list]:
    '''将列表尽量等分为 n 份'''
    size, rest = divmod(len(lst), n)
    start = 0
    for i in range(n):
        step = size + 1 if i < rest else size
        stop = start + step
        yield lst[start:stop]
        start = stop


# TODO: 低版本 shapely 的几何对象
def is_sequence(obj: Any) -> bool:
    '''判断是否为序列'''
    if isinstance(obj, str):
        return False

    try:
        len(obj)
    except Exception:
        return False

    return True


def to_list(obj: Any) -> list:
    '''将对象转为列表'''
    if is_sequence(obj):
        return list(obj)
    return [obj]


class DeprecationError(Exception):
    pass


def deprecator(
    deprecated: Optional[Callable] = None,
    *,
    alternatives: Optional[Union[Callable, Iterable[Callable]]] = None,
    raise_error: bool = False,
) -> Callable:
    '''
    提示函数弃用的装饰器

    Parameters
    ----------
    deprecated : callable, optional
        被弃用的函数。使用装饰器时不需要显式指定该参数。

    alternatives : callable or list of callable, optional
        替代被弃用函数的函数。可以是 None、一个或多个函数。

    raise_error: bool, optional
        是否抛出错误。默认为 False，仅抛出警告。
    '''
    if deprecated is None:
        return partial(
            deprecator, alternatives=alternatives, raise_error=raise_error
        )

    msg = f'{deprecated.__name__} 已弃用'
    if alternatives is not None:
        alternatives = to_list(alternatives)
        sub = '、'.join([func.__name__ for func in alternatives])
        sub = ' 或 '.join(sub.rsplit('、', 1))
        msg += f'，建议换用 {sub}'

    @wraps(deprecated)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if raise_error:
            raise DeprecationError(msg)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        return deprecated(*args, **kwargs)

    return wrapper
