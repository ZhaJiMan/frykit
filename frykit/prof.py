import cProfile
import functools
import time
from collections.abc import Callable
from typing import Any, Literal, Optional, Union

from line_profiler import LineProfiler

from frykit.help import PathType

TimeUnits = Literal['ns', 'us', 'ms', 's', 'min']


def _convert_seconds(seconds, unit: TimeUnits = 's') -> float:
    '''将时间秒数转换为其它单位.'''
    if unit == 'ns':
        scale = 1e9
    if unit == 'us':
        scale = 1e6
    elif unit == 'ms':
        scale = 1e3
    elif unit == 's':
        scale = 1e0
    elif unit == 'min':
        scale = 1 / 60
    else:
        raise ValueError('不支持的单位')

    return seconds * scale


def timer(
    func: Optional[Callable] = None,
    *,
    unit: TimeUnits = 's',
    verbose: bool = True,
    fmt: Optional[str] = None,
    out: Optional[list] = None
) -> Callable:
    '''
    给函数或方法计时的装饰器.

    Parameters
    ----------
    func : callable
        需要被计时的函数或方法.

    unit : {'ns', 'us', 'ms', 's', 'min'}, optional
        时间单位. 默认为s.

    verbose : bool, optional
        是否打印计时结果. 默认为True.

    fmt : str, optional
        打印格式. 默认为'[{name}] {time:.3f} {unit}'.
        其中name对应函数名, time对应耗时, unit对应时间单位.

    out : list, optional
        收集耗时的列表, 装饰器会将耗时添加到列表中. 默认不收集结果.
    '''
    if fmt is None:
        fmt = '[{name}] {time:.3f} {unit}'
    if func is None:
        return functools.partial(
            timer, unit=unit, verbose=verbose, fmt=fmt, out=out
        )

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        dt = _convert_seconds(t1 - t0, unit)

        if verbose:
            info = fmt.format(name=func.__name__, time=dt, unit=unit)
            print(info)
        if out is not None:
            out.append(dt)

        return result

    return wrapper


class Timer:
    '''
    给代码块计时的类.

    Attributes
    ----------
    dt : float or None
        代码块的耗时. 单位由unit决定, 初始值为None.
    '''

    def __init__(
        self,
        unit: TimeUnits = 's',
        verbose: bool = True,
        fmt: Optional[str] = None,
    ):
        '''
        Parameters
        ----------
        unit : {'ns', 'us', 'ms', 's', 'min'}, optional
            时间单位. 默认为s.

        verbose : bool, optional
            是否打印计时结果. 默认为True.

        fmt : str, optional
            打印格式. 默认为'{time:.3f} {unit}'.
            其中time对应耗时, unit对应时间单位.
        '''
        if fmt is None:
            fmt = '{time:.3f} {unit}'
        self.unit = unit
        self.verbose = verbose
        self.fmt = fmt
        self._t0 = None
        self._t1 = None
        self.dt = None

    def start(self) -> None:
        '''开始计时.'''
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        '''停止计时.'''
        self._t1 = time.perf_counter()
        self.dt = _convert_seconds(self._t1 - self._t0, self.unit)
        if self.verbose:
            info = self.fmt.format(time=self.dt, unit=self.unit)
            print(info)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


def cprofiler(filepath: PathType) -> Callable:
    '''cProfile的装饰器. 保存结果到指定路径.'''

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with cProfile.Profile() as profile:
                result = func(*args, **kwargs)
            profile.dump_stats(str(filepath))
            return result

        return wrapper

    return decorator


def lprofiler(filepath: PathType) -> Callable:
    '''line_profiler的装饰器. 保存结果到指定路径.'''

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profile = LineProfiler(func)
            result = profile.runcall(func, *args, **kwargs)
            profile.dump_stats(str(filepath))
            return result

        return wrapper

    return decorator
