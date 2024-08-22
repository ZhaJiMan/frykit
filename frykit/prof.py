import cProfile
import statistics
import time
from collections.abc import Callable
from functools import partial, wraps
from typing import Any, Optional, Union

from frykit._typing import PathType

TimedCallable = Callable[..., Union[float, list[float]]]


def timer(
    func: Optional[Callable] = None,
    *,
    repeat: int = 1,
    number: int = 1,
    mean: bool = True,
) -> Union[TimedCallable, Callable[[Callable], TimedCallable]]:
    '''计时装饰器。使被包装的函数返回 repeat 个运行 number 次的耗时，单位为秒。'''
    if func is None:
        return partial(timer, repeat=repeat, number=number, mean=mean)
    assert repeat > 0 and number > 0

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Union[float, list[float]]:
        dt_list = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            for _ in range(number):
                func(*args, **kwargs)
            t1 = time.perf_counter()
            dt = t1 - t0
            dt_list.append(dt)

        if mean:
            return statistics.mean(dt_list) / number
        if len(dt_list) == 1:
            return dt_list[0]
        return dt_list

    return wrapper


def cprofiler(filepath: PathType) -> Callable:
    '''cProfile 的装饰器。保存结果到指定路径。'''

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with cProfile.Profile() as profile:
                result = func(*args, **kwargs)
            profile.dump_stats(str(filepath))
            return result

        return wrapper

    return decorator


def lprofiler(filepath: PathType) -> Callable:
    '''line_profiler 的装饰器。保存结果到指定路径。'''
    from line_profiler import LineProfiler

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profile = LineProfiler(func)
            result = profile.runcall(func, *args, **kwargs)
            profile.dump_stats(str(filepath))
            return result

        return wrapper

    return decorator
