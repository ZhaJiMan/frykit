from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pandas as pd

from frykit.typing import RealNumber, RealNumberT

__all__ = [
    "arange2",
    "asarrays",
    "az_to_t",
    "binning2d",
    "count_consecutive_trues",
    "dd_to_dm",
    "dd_to_dms",
    "dm_to_dd",
    "dms_to_dd",
    "dms_to_dd",
    "get_values_between",
    "hav",
    "haversine",
    "interp_nearest_2d",
    "interp_nearest_dd",
    "is_finite",
    "lon_to_180",
    "lon_to_360",
    "lonlat_to_xyz",
    "make_circle",
    "make_ellipse",
    "make_evenly_bins",
    "month_to_season",
    "region_mask",
    "rt_to_xy",
    "split_consecutive_trues",
    "t_to_az",
    "uv_to_wswd",
    "wswd_to_uv",
    "xy_to_rt",
]

R90 = np.pi / 2
R180 = np.pi
R270 = 3 * R90
R360 = 2 * R180
R450 = 5 * R90
R540 = 3 * R180


def lon_to_180(lon: ArrayLike, degrees: bool = True) -> NDArray[RealNumber]:
    """经度从 [0, 360] 范围换算到 (-180, 180]，180 会映射到 180。默认使用角度。"""
    lon = np.asarray(lon)
    if degrees:
        return (lon - 540) % -360 + 180
    else:
        return (lon - R540) % -R360 + R180


def lon_to_360(lon: ArrayLike, degrees: bool = True) -> NDArray[RealNumber]:
    """经度从 [-180, 180] 范围换算到 [0, 360)，0 会映射到 0。默认使用角度。"""
    lon = np.asarray(lon)
    return lon % 360 if degrees else lon % R360


def month_to_season(month: ArrayLike) -> NDArray[np.int64]:
    """[1, 12] 范围的月份换算为 [1, 4] 的季节"""
    month = np.asarray(month, dtype=np.int64)
    return (month - 3) % 12 // 3 + 1


def asarrays(*args: ArrayLike, **kwargs: Any) -> list[NDArray[Any]]:
    """对多个参数应用 np.asarray"""
    return list(map(partial(np.asarray, **kwargs), args))


def rt_to_xy(
    r: ArrayLike, t: ArrayLike, degrees: bool = False
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """极坐标转为直角坐标。默认使用弧度。"""
    r, t = asarrays(r, t)
    if degrees:
        t = np.radians(t)
    x = r * np.cos(t)
    y = r * np.sin(t)

    return x, y


def xy_to_rt(
    x: ArrayLike, y: ArrayLike, degrees: bool = False
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """直角坐标转为极坐标。角度范围 (-180, 180]。默认使用弧度。"""
    x, y = asarrays(x, y)
    r = np.hypot(x, y)
    t = np.arctan2(y, x)
    if degrees:
        t = np.degrees(t)

    return r, t


def t_to_az(t: ArrayLike, degrees: bool = False) -> NDArray[RealNumber]:
    """x 轴夹角转为方位角。方位角范围 [0, 360)，90 会映射到 0。默认使用弧度。"""
    t = np.asarray(t)
    if degrees:
        return (90 - t) % 360
    else:
        return (R90 - t) % R360


def az_to_t(az: ArrayLike, degrees: bool = False) -> NDArray[RealNumber]:
    """方位角转为 x 轴夹角。夹角范围 (-180, 180]，270 会映射到 180。默认使用弧度。"""
    az = np.asarray(az)
    if degrees:
        return -(az + 90) % -360 + 180
    else:
        return -(az + R90) % -R360 + R180


def lonlat_to_xyz(
    lon: ArrayLike, lat: ArrayLike, r: float = 1.0, degrees: bool = False
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """经纬度转为球面 xyz 坐标。默认使用弧度。"""
    lon, lat = asarrays(lon, lat)
    if degrees:
        lon = np.radians(lon)
        lat = np.radians(lat)
    cos_lat = np.cos(lat)
    x = r * np.cos(lon) * cos_lat
    y = r * np.sin(lon) * cos_lat
    z = r * np.sin(lat)

    return x, y, z


def wswd_to_uv(
    ws: ArrayLike, wd: ArrayLike, degrees: bool = False
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """风向风速转为 uv。默认使用弧度。"""
    ws, wd = asarrays(ws, wd)
    if degrees:
        wd = np.radians(wd)
    u = -ws * np.sin(wd)
    v = -ws * np.cos(wd)

    return u, v


def uv_to_wswd(
    u: ArrayLike, v: ArrayLike, degrees: bool = False
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """uv 转为风向风速。默认使用弧度。"""
    u, v = asarrays(u, v)
    ws = np.hypot(u, v)
    wd = np.arctan2(u, v) + np.pi
    if degrees:
        wd = np.degrees(wd)

    return ws, wd


def dm_to_dd(d: ArrayLike, m: ArrayLike) -> NDArray[np.floating]:
    """度分转为十进制度数"""
    d, m = asarrays(d, m)
    return d + m / 60


def dms_to_dd(d: ArrayLike, m: ArrayLike, s: ArrayLike) -> NDArray[np.floating]:
    """度分秒转为十进制度数"""
    d, m, s = asarrays(d, m, s)
    return d + m / 60 + s / 3600


def dd_to_dm(dd: ArrayLike) -> tuple[NDArray[RealNumber], NDArray[RealNumber]]:
    """十进制度数转为度分"""
    dd = np.asarray(dd)
    sign = np.sign(dd)
    dd = np.abs(dd)

    d = np.floor(dd)
    m = (dd - d) * 60
    d *= sign

    return d, m


def dd_to_dms(
    dd: ArrayLike,
) -> tuple[NDArray[RealNumber], NDArray[RealNumber], NDArray[RealNumber]]:
    """十进制度数转为度分秒"""
    d, m_ = dd_to_dm(dd)
    m = np.floor(m_)
    s = (m_ - m) * 60

    return d, m, s


def hav(x: ArrayLike) -> NDArray[np.floating]:
    """半正矢函数"""
    return np.square(np.sin(np.asarray(x) / 2))


def haversine(
    lon1: ArrayLike,
    lat1: ArrayLike,
    lon2: ArrayLike,
    lat2: ArrayLike,
    degrees: bool = False,
) -> NDArray[np.floating]:
    """用 haversine 公式计算两点间的圆心角。默认使用弧度。"""
    lon1, lat1, lon2, lat2 = asarrays(lon1, lat1, lon2, lat2)
    if degrees:
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = hav(dlat)
    b = np.cos(lat1) * np.cos(lat2) * hav(dlon)
    dtheta = 2 * np.arcsin(np.sqrt(a + b))
    if degrees:
        dtheta = np.degrees(dtheta)

    return dtheta


def make_ellipse(
    x: float = 0,
    y: float = 0,
    a: float = 1,
    b: float | None = None,
    angle: float = 0,
    npts: int = 100,
    ccw: bool = True,
) -> NDArray[np.float64]:
    """生成椭圆的 xy 坐标序列

    Parameters
    ----------
    x, y : float, default 0
        中心的 xy 坐标。默认为 0。

    a : float, default 1
        半长轴长度。默认为 1。

    b : float or None, default None
        半短轴长度。默认为 None，表示和 a 相等。

    angle : float, default 0
        半长轴和 x 轴的夹角。默认为 0 度。

    npts : int, default 100
        用 np.linspace(0, 2 * pi, npts) 生成 npts 个点。默认为 100。

    ccw : bool, default True
        坐标序列是否沿逆时针方向。默认为 True。

    Returns
    -------
    verts : (npts, 2) ndarray
        xy 坐标序列。最后一个点跟第一个点相同。
    """
    t = np.linspace(0, 2 * np.pi, npts)
    verts = np.column_stack([np.cos(t), np.sin(t), np.ones_like(t)])

    # 对单位圆做仿射变换
    b = a if b is None else b
    angle = math.radians(angle)
    cos = math.cos(angle)
    sin = math.sin(angle)
    mtx = [[a * cos, a * sin], [-b * sin, b * cos], [x, y]]
    verts = verts @ mtx
    if not ccw:
        verts = verts[::-1]

    return verts


def make_circle(
    x: float = 0, y: float = 0, r: float = 1, npts: int = 100, ccw: bool = True
) -> NDArray[np.float64]:
    """生成圆的 xy 坐标序列

    Parameters
    ----------
    x, y : float, default 0
        中心的 xy 坐标。默认为 0。

    r : float, default 1
        圆的半径。默认为 1。

    npts : int, default 100
        用 linspace(0, 2 * pi, npts) 生成 npts 个点。默认为 100。

    ccw : bool, default True
        坐标序列是否沿逆时针方向。默认为 True。

    Returns
    -------
    verts : (npts, 2) ndarray
        xy 坐标序列。最后一个点跟第一个点相同。
    """
    return make_ellipse(x, y, r, npts=npts, ccw=ccw)


def region_mask(
    x: ArrayLike, y: ArrayLike, extents: Sequence[float]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """返回表示坐标点是否落入方框的布尔数组

    Parameters
    ----------
    x, y : array_like
        xy 坐标

    extents : (4,) tuple of float
        方框范围 (x0, x1, y0, y1)


    Returns
    -------
    ndarray or (2,) tuple of ndarray
        x 和 y 的布尔数组，元素为 True 表示落入方框。

    Examples
    --------
    lon_mask, lat_mask = region_mask(lon1d, lat1d, extents)
    lon1d = lon1d[lon_mask]
    lat1d = lat1d[lat_mask]
    ixgrid = np.ix_(lat_mask, lon_mask)
    subset_data2d = data2d[ixgrid]

    lon_mask, lat_mask = region_mask(lon2d, lat2d, extents)
    mask = lon_mask & lat_mask
    subset_data1d = data2d[mask]
    """
    x, y = asarrays(x, y)
    x0, x1, y0, y1 = extents
    return (x >= x0) & (x <= x1), (y >= y0) & (y <= y1)


def count_consecutive_trues(mask: ArrayLike) -> NDArray[np.int64]:
    """统计布尔序列里真值连续出现的次数，返回相同长度的序列。"""
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 1:
        raise ValueError("mask 必须是一维数组")
    if len(mask) == 0:
        return np.array([], dtype=np.int64)

    value_id = np.diff(mask, prepend=mask[0]).cumsum()
    _, unique_counts = np.unique(value_id, return_counts=True)
    value_counts = unique_counts[value_id].astype(np.int64)
    value_counts[~mask] = 0

    return value_counts


def split_consecutive_trues(mask: ArrayLike) -> list[NDArray[np.intp]]:
    """分段返回布尔序列里连续真值段落的索引"""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("mask 必须是一维数组")

    ii = np.nonzero(mask)[0]
    if len(ii) == 0:
        return []
    elif len(ii) == 1:
        return [ii]
    else:
        jj = np.nonzero(np.diff(ii) != 1)[0] + 1
        return np.split(ii, jj)


def interp_nearest_dd(
    points: ArrayLike,
    values: ArrayLike,
    xi: ArrayLike,
    radius: float = float("inf"),
    fill_value: Any = float("nan"),
) -> NDArray[Any]:
    """可以限制搜索半径的多维最近邻插值

    Parameters
    ----------
    points : (n, d) array_like
        数据点的坐标。有 n 个数据点，每个点有 d 个坐标分量

    values : (c, n) array_like
        数据点的变量值。其中通道维度 c 可以省略。

    xi: (m, d) array_like
        插值点的坐标。有 m 个插值点，每个点有 d 个坐标分量。

    radius : float, default inf
        插值点能匹配到数据点的最大距离（半径）。默认为 inf。

    fill_value : default nan
        距离超出 radius 的插值点用 fill_value 填充。默认为 nan。

    Returns
    -------
    result : (c, m) ndarray
        插值结果。当 values 有通道维度 c 时 result 也会有。
    """
    from scipy.spatial import KDTree

    points, xi, values = asarrays(points, xi, values)
    if points.ndim != 2:
        raise ValueError("points 必须是二维数组")
    if xi.ndim != 2:
        raise ValueError("xi 必须是二维数组")
    if values.shape[-1:] != points.shape[:1]:
        raise ValueError("values 的形状跟 points 不匹配")

    tree = KDTree(points)
    dd, ii = tree.query(xi)
    mask = dd > radius

    # 当 fill_value 是字符串时需要避免与 dtype 混淆
    dtype = np.result_type(values, np.asarray(fill_value))
    result = values[..., ii].astype(dtype)
    if mask.any():
        result[..., mask] = fill_value

    return result


def _ravel_stack(arrs: Iterable[NDArray[Any]]) -> NDArray[Any]:
    return np.column_stack(list(map(np.ravel, arrs)))


def interp_nearest_2d(
    x: ArrayLike,
    y: ArrayLike,
    values: ArrayLike,
    xi: ArrayLike,
    yi: ArrayLike,
    radius: float = float("inf"),
    fill_value: Any = float("nan"),
) -> NDArray[Any]:
    """可以限制搜索半径的二维最近邻插值

    相比 interp_nearest_dd，输入的形状更灵活，方便处理卫星的非规则网格数据。

    Parameters
    ----------
    x, y : (n1, n2, ...) array_like
        数据点的坐标

    values : (c, n1, n2, ...) array_like
        数据点的变量值。其中通道维度 c 可以省略。

    xi, yi : (m1, m2, ...) array_like
        插值点的坐标

    radius : float, default inf
        插值点能匹配到数据点的最大距离（半径）。默认为 inf。

    fill_value : default nan
        距离超出 radius 的插值点用 fill_value 填充。默认为 nan。

    Returns
    -------
    result : (c, m1, m2, ...) ndarray
        插值结果。当 values 有通道维度 c 时 result 也会有。
    """
    x, y, xi, yi, values = asarrays(x, y, xi, yi, values)
    if x.ndim == 0:
        raise ValueError("x 至少是一维数组")
    if x.shape != y.shape:
        raise ValueError("要求 x 和 y 形状相同")
    if xi.ndim == 0:
        raise ValueError("xi 至少是一维数组")
    if xi.shape != yi.shape:
        raise ValueError("要求 xi 和 yi 形状相同")
    if values.shape[-x.ndim :] != x.shape:
        raise ValueError("values 的形状跟 x 不匹配")

    # 元组解包能避免出现多余的维度
    channel_shape = values.shape[: -x.ndim]
    return interp_nearest_dd(
        points=_ravel_stack([x, y]),
        values=values.reshape(*channel_shape, -1),
        xi=_ravel_stack([xi, yi]),
        radius=radius,
        fill_value=fill_value,
    ).reshape(*channel_shape, *xi.shape)


def arange2(start: float, stop: float, step: float) -> NDArray[np.float64]:
    """尽可能包含 stop 的 np.arange"""
    length = max((stop - start) // step, 0) + 1
    return np.arange(length, dtype=np.float64) * step + start


def make_evenly_bins(
    start: float, stop: float, step: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """生成等距的一维 bins，并用每个 bin 的中点作为标签。"""
    bins = arange2(start, stop, step)
    if len(bins) < 2:
        raise ValueError("start 和 stop 间至少有一个 step 的长度")
    labels = (bins[:-1] + bins[1:]) / 2

    return bins, labels


def _is_monotonic(a: ArrayLike, ascending: bool = True, strict: bool = False) -> bool:
    """判断一维数组是否单调"""
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("a 必须是一维数组")

    # (0,) 或 (1,) 数组直接返回 True
    if ascending:
        op = np.greater if strict else np.greater_equal
    else:
        op = np.less if strict else np.less_equal
    flag = bool(np.all(op(np.diff(a), 0)))

    return flag


def is_monotonic_increasing(a: ArrayLike, strict: bool = False) -> bool:
    """判断一维数组是否单调递增"""
    return _is_monotonic(a, ascending=True, strict=strict)


def is_monotonic_decreasing(a: ArrayLike, strict: bool = False) -> bool:
    """判断一维数组是否单调递减"""
    return _is_monotonic(a, ascending=False, strict=strict)


def binning2d(
    x: ArrayLike,
    y: ArrayLike,
    values: ArrayLike,
    xbins: ArrayLike,
    ybins: ArrayLike,
    func: str
    | Callable[[pd.Series], Any]
    | Sequence[str | Callable[[pd.Series], Any]] = "mean",
    fill_value: Any = float("nan"),
    right: bool = True,
    include_lowest: bool = False,
) -> NDArray[Any]:
    """对散点数据做二维分箱

    内部通过 pd.cut 实现。当 bins 单调递减时，会先排成升序再调用 pd.cut，最后倒转结果顺序。

    Parameters
    x, y : (n1, n2, ...) array_like
        数据点的坐标

    values : (c, n1, n2, ...) array_like
        数据点的变量值。其中通道维度 c 可以省略。

    xbins : (nx + 1,) array_like
        用于划分 x 的 bins。要求数值单调递增或递减。

    ybins : (ny + 1,) array_like
        用于划分 y 的 bins。要求数值单调递增或递减。

    func : str, callable or (f,) sequence of str and callable, default 'mean'
        对落入 bin 中的数据点做聚合的函数。
        可以是函数名，例如 'count'、'min'、'max'、'mean'、'median' 等；
        或者是输入为至少含一个点的 DataFrame，输出一个标量的函数对象；
        或者是由字符串和函数组成的序列。默认为 'mean'。

    fill_value : default nan
        没有数据点落入的 bin 会用 fill_value 填充。默认为 nan。

    right : bool, default True
        True 时分箱区间右闭合 (x0, x1]；False 时左闭合 [x0, x1)。默认为 True。

    include_lowest: bool, default False
        第一个分箱区间是否包含左边缘。但 right=False 时看不出效果。

    Returns
    -------
    result : (c, f, ny, nx) ndarray
        binning 的结果。当 values 有通道维度 c 时 result 也会有；
        当 func 是序列时 result 会有聚合维度 f。
    """
    import pandas as pd

    def process_bins(bins: NDArray[RealNumberT]) -> tuple[bool, NDArray[RealNumberT]]:
        """检查 bins 有效性，返回是否升序的 flag 和升序的 bins。"""
        if bins.ndim != 1 or len(bins) < 2:
            raise ValueError("bins 必须是长度至少为 2 的一维数组")

        if is_monotonic_increasing(bins):
            return True, bins
        elif is_monotonic_decreasing(bins):
            return False, bins[::-1]
        else:
            raise ValueError("要求 bins 单调递增或递减")

    x, y, values, xbins, ybins = asarrays(x, y, values, xbins, ybins)
    if x.ndim == 0:
        raise ValueError("x 至少是一维数组")
    if x.shape != y.shape:
        raise ValueError("要求 x 和 y 形状相同")
    if values.ndim - x.ndim > 1:
        raise ValueError("values 最多只能有一个通道维度")
    if values.shape[-x.ndim :] != x.shape:
        raise ValueError("values 的形状跟 x 不匹配")

    x_ascending, xbins = process_bins(xbins)
    y_ascending, ybins = process_bins(ybins)
    nx = len(xbins) - 1
    ny = len(ybins) - 1

    # index 用来 reindex 和恢复顺序
    xlabels = np.arange(nx)
    ylabels = np.arange(ny)
    index = pd.MultiIndex.from_product(
        [
            ylabels if y_ascending else ylabels[::-1],
            xlabels if x_ascending else xlabels[::-1],
        ]
    )

    # 一维数组转置形状不变，并且 DataFrame 会正确处理
    channel_shape = values.shape[: -x.ndim]
    df = pd.DataFrame(values.reshape(*channel_shape, -1).T)
    df["x"] = pd.cut(
        x=x.ravel(),
        bins=xbins,
        labels=xlabels,
        right=right,
        include_lowest=include_lowest,
    )
    df["y"] = pd.cut(
        x=y.ravel(),
        bins=ybins,
        labels=ylabels,
        right=right,
        include_lowest=include_lowest,
    )

    result = (
        df.groupby(["y", "x"], observed=True)
        .agg(func)
        .reindex(index, fill_value=fill_value)
        .to_numpy()
    )

    if isinstance(func, str) or callable(func):
        func_shape: tuple[int, ...] = tuple()
    else:
        func_shape = tuple([len(func)])
    result_shape = (*channel_shape, *func_shape, ny, nx)
    result = result.T.reshape(result_shape)

    return result


def is_finite(a: ArrayLike) -> bool:
    """判断数组是否不含 nan 或 inf"""
    return bool(np.isfinite(a).all())


def get_values_between(
    values: ArrayLike, vmin: float, vmax: float
) -> NDArray[RealNumber]:
    """获取 vmin <= values <= vmax 的元素"""
    values = np.asarray(values)
    return values[(values >= vmin) & (values <= vmax)].copy()
