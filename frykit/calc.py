import math
import re
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

R90 = np.pi / 2
R180 = np.pi
R270 = 3 * R90
R360 = 2 * R180
R450 = 5 * R90
R540 = 3 * R180


def _asarrays(*args: ArrayLike, **kwargs: Any) -> list[NDArray]:
    '''对多个参数应用 np.asarray'''
    return list(map(partial(np.asarray, **kwargs), args))


def lon_to_180(lon: ArrayLike, degrees: bool = True) -> NDArray:
    '''将经度换算到 (-180, 180] 范围内。默认使用角度。'''
    lon = np.asarray(lon)
    if degrees:
        return (lon - 540) % -360 + 180
    else:
        return (lon - R540) % -R360 + R180


def lon_to_360(lon: ArrayLike, degrees: bool = True) -> NDArray:
    '''将经度换算到 [0, 360) 范围内。默认使用角度。'''
    lon = np.asarray(lon)
    if degrees:
        return lon % 360
    else:
        return lon % R360


def month_to_season(month: ArrayLike) -> NDArray:
    '''将月份换算为季节。月份用 [1, 12] 表示，季节用 [1, 4] 表示。'''
    month = np.asarray(month)
    assert issubclass(month.dtype.type, np.integer)
    return (month - 3) % 12 // 3 + 1


def rt_to_xy(
    r: ArrayLike, t: ArrayLike, degrees: bool = False
) -> tuple[NDArray, NDArray]:
    '''极坐标转直角坐标。默认使用弧度。'''
    r, t = _asarrays(r, t)
    if degrees:
        t = np.radians(t)
    x = r * np.cos(t)
    y = r * np.sin(t)

    return x, y


def xy_to_rt(
    x: ArrayLike, y: ArrayLike, degrees: bool = False
) -> tuple[NDArray, NDArray]:
    '''直角坐标转极坐标。默认使用弧度，角度范围 (-180, 180]。'''
    x, y = _asarrays(x, y)
    r = np.hypot(x, y)
    t = np.arctan2(y, x)
    if degrees:
        t = np.degrees(t)

    return r, t


def t_to_az(t: ArrayLike, degrees: bool = False) -> NDArray:
    '''x 轴夹角转方位角。默认使用弧度，az 范围 [0, 360)。'''
    t = np.asarray(t)
    if degrees:
        return (90 - t) % 360
    else:
        return (R90 - t) % R360


def az_to_t(az: ArrayLike, degrees: bool = False) -> NDArray:
    '''方位角转 x 轴夹角。默认使用弧度，t 范围 (-180, 180]。'''
    az = np.asarray(az)
    if degrees:
        return -(az + 90) % -360 + 180
    else:
        return -(az + R90) % -R360 + R180


def lon_lat_to_xyz(
    lon: ArrayLike, lat: ArrayLike, r: float = 1.0, degrees: bool = False
) -> tuple[NDArray, NDArray, NDArray]:
    '''经纬度转为球面 xyz 坐标。默认使用弧度。'''
    lon, lat = _asarrays(lon, lat)
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
) -> tuple[NDArray, NDArray]:
    '''风向风速转为 uv。默认使用弧度。'''
    ws, wd = _asarrays(ws, wd)
    if degrees:
        wd = np.radians(wd)
    u = -ws * np.sin(wd)
    v = -ws * np.cos(wd)

    return u, v


def uv_to_wswd(
    u: ArrayLike, v: ArrayLike, degrees: bool = False
) -> tuple[NDArray, NDArray]:
    '''uv 转为风向风速。默认使用弧度。'''
    u, v = _asarrays(u, v)
    ws = np.hypot(u, v)
    wd = np.arctan2(u, v) + np.pi
    if degrees:
        wd = np.degrees(wd)

    return ws, wd


def _hms_to_degrees(hour: float, minute: float, second: float) -> float:
    return hour + minute / 60 + second / 3600


def hms_to_degrees(
    hour: ArrayLike, minute: ArrayLike, second: ArrayLike
) -> NDArray:
    '''时分秒转为度数'''
    hour, minute, second = _asarrays(hour, minute, second)
    return _hms_to_degrees(hour, minute, second)  # type: ignore


def _split_hms(hms: str) -> tuple[float, float, float]:
    hour, minute, second = map(float, re.split(r'[^\d.]+', hms)[:3])
    return hour, minute, second


def hms_to_degrees2(hms: Union[str, Iterable[str]]) -> Union[float, NDArray]:
    '''时分秒转为度数。要求 hms 是形如 43°08′20″ 的字符串。'''
    if isinstance(hms, str):
        return _hms_to_degrees(*_split_hms(hms))
    return np.array(list(map(hms_to_degrees2, hms)))


def hav(x: ArrayLike) -> NDArray:
    '''半正矢函数'''
    x = np.asarray(x)
    return np.sin(x / 2) ** 2


def haversine(
    lon1: ArrayLike,
    lat1: ArrayLike,
    lon2: ArrayLike,
    lat2: ArrayLike,
    degrees: bool = False,
) -> NDArray:
    '''利用 haversine 公式计算两点间的圆心角'''
    lon1, lat1, lon2, lat2 = _asarrays(lon1, lat1, lon2, lat2)
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
    b: Optional[float] = None,
    angle: float = 0,
    npts: int = 100,
    ccw: bool = True,
) -> NDArray:
    '''
    生成椭圆的 xy 坐标序列

    Parameters
    ----------
    x, y : float, optional
        中心的横纵坐标。默认为 (0, 0)。

    a : float, optional
        半长轴长度。默认为 1。

    b : float, optional
        半短轴长度。默认为 None，表示和 a 相等。

    angle : float, optional
        半长轴和 x 轴成的角度。默认为 0 度。

    npts : int, optional
        用 linspace(0, 2 * pi, npts) 生成 npts 个角度。
        默认为 100，要求大于等于 4。

    ccw : bool, optional
        坐标序列是否沿逆时针方向。默认为 True。

    Returns
    -------
    verts : (npts, 2) ndarray
        xy 坐标序列
    '''
    if npts < 4:
        raise ValueError('npts < 4')
    t = np.linspace(0, 2 * np.pi, npts)
    verts = np.c_[np.cos(t), np.sin(t), np.ones_like(t)]

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
) -> NDArray:
    '''
    生成圆的 xy 坐标序列

    Parameters
    ----------
    x, y : float, optional
        中心的横纵坐标。默认为 (0, 0)。

    r : float, optional
        圆的半径。默认为 1。

    npts : int, optional
        用 linspace(0, 2 * pi, npts) 生成 npts 个角度。
        默认为 100，要求大于等于 4。

    ccw : bool, optional
        坐标序列是否沿逆时针方向。默认为 True。

    Returns
    -------
    verts : (npts, 2) ndarray
        xy 坐标序列
    '''
    return make_ellipse(x, y, r, npts=npts, ccw=ccw)


def region_ind(
    lon: ArrayLike,
    lat: ArrayLike,
    extents: tuple[float, float, float, float],
    form: Literal['mask', 'ix'] = 'mask',
) -> Union[NDArray, tuple[NDArray, NDArray]]:
    '''
    返回落入给定经纬度方框范围内的索引

    Parameters
    ----------
    lon : array_like
        经度。若 form='mask' 则要求形状与 lat 一致。

    lat : array_like
        纬度。若 form='mask' 则要求形状与 lon 一致。

    extents : (4,) tuple of float
        经纬度方框的范围 [lon0, lon1, lat0, lat1]

    form : {'mask', 'ix'}
        索引的形式。
        'mask'：使用与 lon 和 lat 同形状的布尔数组进行索引。
        'ix'：使用下标构成的开放网格进行索引。

    Returns
    -------
    ind : ndarray or 2-tuple of ndarray
        索引的布尔数组，或者两个下标数组构成的元组。

    Examples
    --------
    mask = region_ind(lon2d, lat2d, extents)
    data1d = data2d[mask]

    ixgrid = region_ind(lon1d, lat1d, extents, form='ix')
    data2d_subset = data2d[ixgrid]
    data3d_subset = data3d[:, ixgrid[0], ixgrid[1]]
    '''
    lon, lat = _asarrays(lon, lat)
    lon0, lon1, lat0, lat1 = extents
    lon_mask = (lon >= lon0) & (lon <= lon1)
    lat_mask = (lat >= lat0) & (lat <= lat1)
    if form == 'mask':
        if lon.shape != lat.shape:
            raise ValueError('lon 和 lat 的形状应该一样')
        ind = lon_mask & lat_mask
    elif form == 'ix':
        ind = np.ix_(lon_mask, lat_mask)
    else:
        raise ValueError("form: {'mask', 'ix'}")

    return ind  # type: ignore


def count_consecutive_values(
    series: ArrayLike, cond: Union[ArrayLike, Callable[[NDArray], NDArray]]
) -> NDArray:
    '''
    统计序列中满足条件的元素连续出现的次数

    Parameters
    ----------
    series : (n,) array_like
        一维序列

    cond : (n,) array_like of bool or callable
        筛选特定元素的条件。可以是与 series 对应的布尔数组，
        也可以是 callable，接受 np.asarray(series) 并返回布尔数组。
        用法类似 DataFrame.where 的 cond 参数。

    Returns
    -------
    value_counts : (n,) ndarray of int
        满足条件的位置的值为元素连续出现的次数，不满足条件的位置的值为 0。

    Examples
    --------
    series = [0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6]
    counts = count_consecutive_values(series, lambda x: x == 0)
    series[counts >= 3] = np.nan
    '''
    series = np.asarray(series)
    assert series.ndim == 1

    if callable(cond):
        cond = cond(series)
    cond = np.asarray(cond, dtype=bool)
    assert cond.shape == series.shape

    if len(series) == 0:
        return np.array([], dtype=int)

    value_id = np.r_[0, np.diff(cond).cumsum()]
    unique, unique_counts = np.unique(value_id, return_counts=True)
    value_counts = unique_counts[np.searchsorted(unique, value_id)]
    value_counts[~cond] = 0

    return value_counts


def count_consecutive_non_positives(series: ArrayLike) -> NDArray:
    '''统计连续出现非正数的次数'''
    return count_consecutive_values(series, cond=lambda x: x <= 0)


def count_consecutive_nans(series: ArrayLike) -> NDArray:
    '''统计连续出现 NaN 的次数'''
    return count_consecutive_values(series, np.isnan)


def split_mask(mask: ArrayLike) -> list[NDArray]:
    '''分段返回布尔数组里连续真值段落的索引'''
    mask = np.asarray(mask, dtype=bool)
    assert mask.ndim == 1

    ii = np.nonzero(mask)[0]
    n = len(ii)
    if n == 0:
        return []
    elif n == 1:
        return [ii]
    else:
        jj = np.nonzero(np.diff(ii) != 1)[0] + 1
        return np.split(ii, jj)


def interp_nearest_dd(
    points: ArrayLike,
    values: ArrayLike,
    xi: ArrayLike,
    radius: float = np.inf,
    fill_value: float = np.nan,
) -> NDArray:
    '''
    可以限制搜索半径的多维最近邻插值

    Parameters
    ----------
    points : (n, d) array_like
        n 个数据点的坐标。每个点有 d 个坐标分量。

    values : (n, ...) array_like
        n 个数据点的变量值。可以有多个波段。

    xi: (m, d) array_like
        m 个插值点的坐标。每个点有 d 个坐标分量。

    radius : float, optional
        插值点能匹配到数据点的最大距离（半径）。默认为 Inf。

    fill_value : float, optional
        距离超出 radius 的插值点用 fill_value 填充。默认为 NaN。

    Returns
    -------
    result : (m, ...) ndarray
        插值点处变量值的数组（浮点型）
    '''
    points, xi = _asarrays(points, xi)
    values = np.asarray(values, dtype=float)
    if points.ndim != 2:
        raise ValueError('points 的维度应该为 2')
    if xi.ndim != 2:
        raise ValueError('xi 的维度应该为 2')
    if values.shape[0] != points.shape[0]:
        raise ValueError('values 和 points 的形状不匹配')

    from scipy.spatial import KDTree

    # 利用 KDTree 搜索最近点
    tree = KDTree(points)
    dd, ii = tree.query(xi)
    result = values[ii]
    result[dd > radius] = fill_value

    return result


def interp_nearest_2d(
    x: ArrayLike,
    y: ArrayLike,
    values: ArrayLike,
    xi: ArrayLike,
    yi: ArrayLike,
    radius: float = np.inf,
    fill_value: float = np.nan,
) -> NDArray:
    '''
    可以限制搜索半径的二维最近邻插值

    Parameters
    ----------
    x : array_like
        数据点的横坐标。要求形状与 y 相同。

    y : array_like
        数据点的纵坐标。要求形状与 x 相同。

    values : array_like
        数据点的变量值。可以有多个波段，要求前面维度的形状与 x 相同。

    xi: array_like
        插值点的横坐标。要求形状与 yi 相同。

    yi: array_like
        插值点的纵坐标。要求形状与 xi 相同。

    radius : float, optional
        插值点能匹配到数据点的最大距离（半径）。默认为 Inf。

    fill_value : float, optional
        距离超出 radius 的插值点用 fill_value 填充。默认为 NaN。

    Returns
    -------
    result : ndarray
        插值点处变量值的数组（浮点型）。
    '''
    x, y, xi, yi, values = _asarrays(x, y, xi, yi, values)
    if x.shape != y.shape:
        raise ValueError('x 和 y 的形状应该一样')
    if values.shape[: x.ndim] != x.shape:
        raise ValueError('values 和 x 的形状不匹配')
    if xi.shape != yi.shape:
        raise ValueError('xi 和 yi 的形状应该一样')

    # 元组解包能避免出现多余的维度
    band_shape = values.shape[x.ndim :]
    return interp_nearest_dd(
        points=np.c_[x.ravel(), y.ravel()],
        values=values.reshape(-1, *band_shape),
        xi=np.c_[xi.ravel(), yi.ravel()],
        radius=radius,
        fill_value=fill_value,
    ).reshape(*xi.shape, *band_shape)


def binned_average_2d(
    x: ArrayLike,
    y: ArrayLike,
    values: Union[ArrayLike, list[ArrayLike]],
    xbins: ArrayLike,
    ybins: ArrayLike,
) -> tuple[NDArray, NDArray, NDArray]:
    '''
    用平均的方式对数据做二维 binning

    Parameters
    ----------
    x : (n,) array_like
        数据点的横坐标

    y : (n,) array_like
        数据点的纵坐标

    values : (n,) array_like or (m,) list of array_like
        数据点的变量值，也可以是一组变量。

    xbins : (nx + 1,) array_like
        用于划分横坐标的 bins

    ybins : (ny + 1,) array_like
        用于划分纵坐标的 bins

    Returns
    -------
    xc : (nx,) ndarray
        xbins 每个 bin 中心的横坐标

    yc : (ny,) ndarray
        ybins 每个 bin 中心的纵坐标

    avg : (ny, nx) or (m, ny, nx) ndarray
        xbins 和 ybins 构成的网格内的变量平均值。
        为了便于画图，采取 (ny, nx) 的形状。
    '''

    def nanmean(arr: NDArray) -> float:
        '''避免空切片警告的 nanmean'''
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.nan
        return arr.mean()

    from scipy.stats import binned_statistic_2d

    # 参数检查由 scipy 的函数负责。注意 x 和 y 的顺序。
    avg, ybins, xbins, _ = binned_statistic_2d(
        y, x, values, bins=[ybins, xbins], statistic=nanmean
    )

    xc = (xbins[1:] + xbins[:-1]) / 2
    yc = (ybins[1:] + ybins[:-1]) / 2

    return xc, yc, avg
