from __future__ import annotations

import cProfile
from collections.abc import Callable
from functools import wraps

from line_profiler import LineProfiler

from frykit.typing import P, StrOrBytesPath, T

__all__ = ["cprofiler", "lprofiler"]


def cprofiler(filepath: StrOrBytesPath) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """cProfile 的装饰器。保存结果到指定路径。"""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with cProfile.Profile() as profile:
                result = func(*args, **kwargs)
            profile.dump_stats(str(filepath))
            return result

        return wrapper

    return decorator


def lprofiler(filepath: StrOrBytesPath) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """line_profiler 的装饰器。保存结果到指定路径。"""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            profile = LineProfiler(func)
            result = profile.runcall(func, *args, **kwargs)
            profile.dump_stats(str(filepath))
            return result

        return wrapper

    return decorator
