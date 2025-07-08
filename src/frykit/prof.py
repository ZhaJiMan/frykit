from __future__ import annotations

import cProfile
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from line_profiler import LineProfiler

from frykit.typing import PathType
from frykit.utils import deprecator

F = TypeVar("F", bound=Callable)


def cprofiler(filepath: PathType) -> Callable[[F], F]:
    """cProfile 的装饰器。保存结果到指定路径。"""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with cProfile.Profile() as profile:
                result = func(*args, **kwargs)
            profile.dump_stats(str(filepath))
            return result

        return cast(F, wrapper)

    return decorator


def lprofiler(filepath: PathType) -> Callable[[F], F]:
    """line_profiler 的装饰器。保存结果到指定路径。"""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profile = LineProfiler(func)
            result = profile.runcall(func, *args, **kwargs)
            profile.dump_stats(str(filepath))
            return result

        return cast(F, wrapper)

    return decorator


@deprecator(raise_error=True)
def timer(*args, **kwargs): ...
