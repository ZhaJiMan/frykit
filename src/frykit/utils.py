from __future__ import annotations

import os
import shutil
import warnings
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol, cast, overload

from frykit.typing import F, HashableT, PathType, T

__all__ = [
    "DeprecationError",
    "HasFullName",
    "chdir_context",
    "compare_sets",
    "del_dir",
    "deprecator",
    "format_literal_error",
    "format_type_error",
    "get_full_name",
    "get_package_version",
    "join_with_cn_comma",
    "new_dir",
    "renew_dir",
    "simple_deprecator",
    "split_list",
    "to_list",
]


def new_dir(dir_path: PathType) -> Path:
    """新建目录"""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path


def del_dir(dir_path: PathType) -> Path:
    """删除目录"""
    dir_path = Path(dir_path)
    if dir_path.exists():
        shutil.rmtree(dir_path)

    return dir_path


def renew_dir(dir_path: PathType) -> Path:
    """重建目录"""
    return new_dir(del_dir(dir_path))


@contextmanager
def chdir_context(dir_path: PathType) -> Generator[None]:
    """临时切换工作目录的上下文管理器"""
    cwd = Path.cwd()
    try:
        os.chdir(dir_path)
        yield
    finally:
        os.chdir(cwd)


def get_package_version(name: str) -> tuple[int, ...]:
    """获取包的版本号"""
    return tuple(map(int, version(name).split(".")))


def split_list(lst: list[T], n: int) -> Iterator[list[T]]:
    """列表尽量等分为 n 份"""
    size, rest = divmod(len(lst), n)
    start = 0
    for i in range(n):
        step = size + 1 if i < rest else size
        stop = start + step
        yield lst[start:stop]
        start = stop


def to_list(obj: Any) -> list[Any]:
    """可迭代对象转为列表，非可迭代对象和字符串用 list 包装。"""
    if isinstance(obj, str):
        return [obj]

    try:
        iter(obj)
        return list(obj)
    except TypeError:
        return [obj]


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
    """用中文顿号和或字连接一组字符串"""
    return " 或 ".join("、".join(strings).rsplit("、", 1))


class HasFullName(Protocol):
    __module__: str
    __qualname__: str


def get_full_name(obj: HasFullName) -> str:
    # 类方法无法用 runtime_checkable 检查
    assert hasattr(obj, "__module__") and hasattr(obj, "__qualname__")
    if obj.__module__ in {"__main__", "builtins"}:
        return obj.__qualname__
    else:
        return f"{obj.__module__}.{obj.__qualname__}"


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
        可以是字符串、类型，或一组字符串和类型。字符串用来表示一些不方便表示的类型。

    Returns
    -------
    msg : str
        消息字符串
    """
    names = []
    for typ in to_list(expected_type):
        match typ:
            case str():
                names.append(typ)
            case type():
                names.append(get_full_name(typ))
            case _:
                raise TypeError(format_type_error("expected_type", typ, [str, type]))

    if len(names) == 0:
        raise ValueError("expected_type 不能为空")

    expected_type_str = join_with_cn_comma(names)
    actual_type_str = get_full_name(type(param_value))
    msg = f"{param_name} 必须是 {expected_type_str} 类型，但传入的是 {actual_type_str} 类型"

    return msg


def format_literal_error(
    param_name: str, param_value: Any, literal_value: Any | Iterable[Any]
) -> str:
    """
    构造用于字面值 ValueError 的消息字符串

    Parameters
    ----------
    param_name : str
        参数名

    param_value
        参数值

    literal_value
        要求的字面值。可以是一组字面值。

    Returns
    -------
    msg : str
        消息字符串
    """
    literal_values = to_list(literal_value)
    if len(literal_values) == 0:
        raise ValueError("literal_value 不能为空")
    if len(literal_values) != len(set(literal_values)):
        raise ValueError("literal_value 不能包含重复的元素")

    param_value_str = repr(param_value)
    literal_value_str = "{" + ", ".join(map(repr, literal_values)) + "}"
    msg = f"{param_name} 只能是 {literal_value_str} 中的一项，但传入的是 {param_value_str}"

    return msg


class DeprecationError(Exception):
    pass


@overload
def deprecator(
    deprecated: F,
    *,
    alternative: str | Callable | Iterable[str | Callable] | None = None,
    raise_error: bool = False,
) -> F: ...


@overload
def deprecator(
    deprecated: None = None,
    *,
    alternative: str | Callable | Iterable[str | Callable] | None = None,
    raise_error: bool = False,
) -> Callable[[F], F]: ...


# TODO: 类装饰器
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
    deprecated : callable or None, default None
        被弃用的函数。默认为 None，表示返回一个带关键字参数的装饰器。

    alternative : str, callable or iterable object of str and callable or None, default None
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

        def decorator(deprecated: F) -> F:
            return deprecator(
                deprecated, alternative=alternative, raise_error=raise_error
            )

        return decorator

    msg = f"{get_full_name(deprecated)} 已弃用"
    if alternative is not None:
        names = []
        for func in to_list(alternative):
            if isinstance(func, str):
                names.append(func)
            elif callable(func):
                names.append(get_full_name(func))
            else:
                raise TypeError(
                    format_type_error("alternative", func, [str, "callable"])
                )

        if len(names) == 0:
            raise ValueError("alternative 不能为空")
        msg += f"，建议换用 {join_with_cn_comma(names)}"

    @wraps(deprecated)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if raise_error:
            raise DeprecationError(msg)
        else:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return deprecated(*args, **kwargs)

    return wrapper


def simple_deprecator(reason: str, raise_error: bool = False) -> Callable[[F], F]:
    """
    提示函数弃用的装饰器

    Parameters
    ----------
    reason : str
        弃用原因。可以用 {name} 占位符表示被弃用的函数名。

    raise_error : bool, default False
        是否抛出错误。默认为 False，表示仅抛出警告。

    Returns
    -------
    callable
        以被弃用函数为参数的装饰器
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            msg = reason.format(name=get_full_name(func))
            if raise_error:
                raise DeprecationError(msg)
            else:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
