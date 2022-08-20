import time
import cProfile
import functools

from line_profiler import LineProfiler

def timer(func=None, *, prompt=True, fmt=None, prec=None, out=None):
    '''
    计时用的装饰器.

    Parameters
    ----------
    func : callable
        需要被计时的函数或方法.

    prompt : bool
        是否打印计时结果.

    fmt : str
        打印格式. %n表示函数名, %t表示耗时.

    prec : int
        打印时的小数位数.

    out : list
        收集耗时的列表.
    '''
    if fmt is None:
        fmt = '[%n] %t s'
    if func is None:
        return functools.partial(
            timer, prompt=prompt, fmt=fmt, prec=prec, out=out
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        dt = t1 - t0

        # 是否打印结果.
        if prompt:
            st = str(dt) if prec is None else f'{dt:.{prec}f}'
            info = fmt.replace('%n', func.__name__).replace('%t', st)
            print(info)

        # 是否输出到列表中.
        if out is not None:
            out.append(dt)

        return result

    return wrapper

def cprofiler(filename):
    '''cProfile的装饰器. 保存结果到指定路径.'''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with cProfile.Profile() as profile:
                result = func(*args, **kwargs)
            profile.dump_stats(filename)
            return result
        return wrapper
    return decorator

def lprofiler(filename):
    '''line_profiler的装饰器. 保存结果到指定路径.'''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profile = LineProfiler(func)
            result = profile.runcall(func, *args, **kwargs)
            profile.dump_stats(filename)
            return result
        return wrapper
    return decorator