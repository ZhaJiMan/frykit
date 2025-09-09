from __future__ import annotations

import cProfile
from collections.abc import Callable
from functools import wraps
from typing import Any, cast

from line_profiler import LineProfiler

from frykit.typing import F, PathType

__all__ = ["cprofiler", "lprofiler"]


def cprofiler(file_path: PathType) -> Callable[[F], F]:
    """cProfile 的装饰器。保存结果到指定路径。"""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with cProfile.Profile() as profile:
                result = func(*args, **kwargs)
            profile.dump_stats(str(file_path))
            return result

        return cast(F, wrapper)

    return decorator


def lprofiler(file_path: PathType) -> Callable[[F], F]:
    """line_profiler 的装饰器。保存结果到指定路径。"""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profile = LineProfiler(func)
            result = profile.runcall(func, *args, **kwargs)
            profile.dump_stats(str(file_path))
            return result

        return cast(F, wrapper)

    return decorator
