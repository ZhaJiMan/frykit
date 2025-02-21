from __future__ import annotations

import shutil
import warnings
from collections.abc import Callable, Hashable, Iterable, Iterator
from functools import partial, wraps
from pathlib import Path
from typing import Any, Protocol, TypeVar, overload

from frykit.typing import PathType


def new_dir(dirpath: PathType) -> Path:
    """新建目录"""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    return dirpath


def del_dir(dirpath: PathType) -> Path:
    """删除目录"""
    dirpath = Path(dirpath)
    if dirpath.exists():
        shutil.rmtree(dirpath)

    return dirpath


def renew_dir(dirpath: PathType) -> Path:
    """重建目录"""
    dirpath = Path(dirpath)
    if dirpath.exists():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True)

    return dirpath


T = TypeVar("T")


def split_list(lst: list[T], n: int) -> Iterator[list[T]]:
    """将列表尽量等分为 n 份"""
    size, rest = divmod(len(lst), n)
    start = 0
    for i in range(n):
        step = size + 1 if i < rest else size
        stop = start + step
        yield lst[start:stop]
        start = stop


# https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
def is_iterable(obj: Any, include_str: bool = True) -> bool:
    """对象是否可迭代"""
    if isinstance(obj, str):
        return include_str

    try:
        iter(obj)
        return True
    except TypeError:
        return False


@overload
def as_list(obj: str) -> list[str]: ...


@overload
def as_list(obj: Iterable[T]) -> list[T]: ...


@overload
def as_list(obj: T) -> list[T]: ...


def as_list(obj: Any) -> list:
    """将对象转为列表"""
    if is_iterable(obj, include_str=False):
        return list(obj)
    else:
        return [obj]


HashableT = TypeVar("HashableT", bound=Hashable)


def compare_sets(
    old_values: Iterable[HashableT], new_values: Iterable[HashableT]
) -> tuple[set[HashableT], set[HashableT]]:
    """将新旧两组值转换为集合，比较得到新增的元素和缺失的元素。"""
    old_set = set(old_values)
    new_set = set(new_values)
    added_set = new_set - old_set
    removed_set = old_set - new_set

    return added_set, removed_set


def join_with_cn_comma(strings: Iterable[str]) -> str:
    """将一组字符串用中文顿号和或字连接"""
    return " 或 ".join("、".join(strings).rsplit("、", 1))


class HasFullName(Protocol):
    __module__: str
    __qualname__: str


def _get_full_name(obj: HasFullName) -> str:
    """获取对象的 {__module__}.{__qualname__}"""
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if module is None or qualname is None:
        raise ValueError("obj 的 __module__ 和 __qualname__ 属性不能为 None")

    if module in {"__main__", "builtins"}:
        return qualname
    else:
        return f"{module}.{qualname}"


def format_type_error(
    param_name: str, param_value: Any, expected_type: str | type | Iterable[str | type]
) -> str:
    """
    构造用于 TypeError 的消息字符串

    Parameters
    ----------
    param_name : str
        参数名

    param_value
        参数值。其类型会反应在消息里

    expected_type: str, type or iterable object of str and type
        期望参数值是什么类型，在消息中用来表示 param_value 的类型与期望不符。
        可以是字符串、type，或一组字符串和 type。字符串用来表示一些不方便表示的类型。

    Returns
    -------
    msg : str
        消息字符串
    """
    names = []
    for typ in as_list(expected_type):
        if isinstance(typ, str):
            names.append(typ)
        elif isinstance(typ, type):
            names.append(_get_full_name(typ))
        else:
            raise TypeError(format_type_error("expected_type", typ, [str, type]))

    expected_type_str = join_with_cn_comma(names)
    actual_type_str = _get_full_name(type(param_value))
    msg = f"{param_name} 必须是 {expected_type_str} 类型，但传入的是 {actual_type_str} 类型"

    return msg


F = TypeVar("F", bound=Callable[..., Any])


class DeprecationError(Exception):
    pass


@overload
def deprecator(
    deprecated: None = None,
    *,
    alternative: str | Callable | Iterable[str | Callable] | None = None,
    raise_error: bool = False,
) -> Callable[[F], F]: ...


@overload
def deprecator(
    deprecated: F,
    *,
    alternative: str | Callable | Iterable[str | Callable] | None = None,
    raise_error: bool = False,
) -> F: ...


# https://python3-cookbook.readthedocs.io/zh-cn/latest/c09/p06_define_decorator_that_takes_optional_argument.html
def deprecator(
    deprecated: F | None = None,
    *,
    alternative: str | Callable | Iterable[str | Callable] | None = None,
    raise_error: bool = False,
) -> F | Callable[[F], F]:
    """
    提示函数弃用的装饰器

    Parameters
    ----------
    deprecated : callable, default None
        被弃用的函数。默认为 None，表示返回一个带关键字参数的装饰器。

    alternative : str, callable or iterable object of str and callable, default None
        建议换用的函数。可以是字符串、函数对象，或一组字符串和函数对象。
        字符串用来表示不方便表示的函数的名称。
        默认为 None，表示被弃用的函数无需替代。

    raise_error: bool, default False
        是否抛出错误。默认为 False，表示仅抛出警告。

    Returns
    -------
    callable
        当 deprecated 为 None 时返回装饰器，不为 None 时返回 wrapper 函数。
    """
    if deprecated is None:
        return partial(deprecator, alternative=alternative, raise_error=raise_error)

    msg = f"{_get_full_name(deprecated)} 已弃用"
    if alternative is not None:
        names = []
        for func in as_list(alternative):
            if isinstance(func, str):
                names.append(func)
            elif callable(func):
                names.append(_get_full_name(func))
            else:
                raise TypeError(
                    format_type_error("alternative", func, [str, "callable"])
                )
        msg += f"，建议换用 {join_with_cn_comma(names)}"

    @wraps(deprecated)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if raise_error:
            raise DeprecationError(msg)
        else:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return deprecated(*args, **kwargs)

    return wrapper
