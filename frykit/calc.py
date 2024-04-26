import math
import re
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import numpy as np


def lon_to_180(lon: Any) -> Any:
    '''将经度换算到(-180, 180]范围内.'''
    return (lon - 540) % -360 + 180


def lon_to_360(lon: Any) -> Any:
    '''将经度换算到[0, 360)范围内.'''
    return lon % 360


def month_to_season(month: Any) -> Any:
    '''将月份换算为季节. 月份用[1, 12]表示, 季节用[1, 4]表示.'''
    return (month - 3) % 12 // 3 + 1


def rt_to_xy(r: Any, t: Any, degrees: bool = False) -> tuple[Any, Any]:
    '''极坐标转直角坐标. 默认使用弧度.'''
    if degrees:
        t = np.radians(t)
    x = r * np.cos(t)
    y = r * np.sin(t)

    return x, y


def xy_to_rt(x: Any, y: Any, degrees: bool = False) -> tuple[Any, Any]:
    '''直角坐标转极坐标. 默认使用弧度, 角度范围(-180, 180].'''
    r = np.hypot(x, y)
    t = np.arctan2(y, x)
    if degrees:
        t = np.degrees(t)

    return r, t


def t_to_az(t: Any, degrees: bool = False) -> Any:
    '''x轴夹角转方位角. 默认使用弧度, az范围[0, 360).'''
    if degrees:
        az = (90 - t) % 360
    else:
        _90 = np.pi / 2
        _360 = 4 * _90
        az = (_90 - t) % _360

    return az


def az_to_t(az: Any, degrees: bool = False) -> Any:
    '''方位角转x轴夹角. 默认使用弧度, t范围(-180, 180].'''
    if degrees:
        t = -(az + 90) % -360 + 180
    else:
        _90 = np.pi / 2
        _180 = 2 * _90
        _360 = 4 * _90
        t = -(az + _90) % -_360 + _180

    return t


def lon_lat_to_xyz(lon: Any, lat: Any, r=1.0, degrees: bool = False) -> Any:
    '''经纬度转为球面xyz坐标. 默认使用弧度.'''
    if degrees:
        lon = np.radians(lon)
        lat = np.radians(lat)
    cos_lat = np.cos(lat)
    x = r * np.cos(lon) * cos_lat
    y = r * np.sin(lon) * cos_lat
    z = r * np.sin(lat)

    return x, y, z


def wswd_to_uv(ws: Any, wd: Any, degrees: bool = False) -> tuple[Any, Any]:
    '''风向风速转为uv. 默认使用弧度.'''
    if degrees:
        wd = np.radians(wd)
    u = -ws * np.sin(wd)
    v = -ws * np.cos(wd)

    return u, v


def uv_to_wswd(u: Any, v: Any, degrees: bool = False) -> tuple[Any, Any]:
    '''uv转为风向风速. 默认使用弧度.'''
    ws = np.hypot(u, v)
    wd = np.arctan2(u, v) + np.pi
    if degrees:
        wd = np.degrees(wd)

    return ws, wd


def hms_to_degrees(hour: Any, minute: Any, second: Any) -> Any:
    '''时分秒转为度数.'''
    return hour + minute / 60 + second / 3600


def hms_to_degrees2(hms: Union[str, Sequence[str]]) -> list[float]:
    '''时分秒转为度数. 要求hms是形如43°08′20″的字符串.'''

    def func(string: str) -> tuple[float, float, float]:
        return map(float, re.split(r'[^\d.]+', string)[:3])

    if isinstance(hms, str):
        return hms_to_degrees(*func(hms))
    return list(map(func, hms))


def hav(x: Any) -> Any:
    '''半正矢函数.'''
    return np.sin(x / 2) ** 2


def haversine(
    lon1: Any, lat1: Any, lon2: Any, lat2: Any, degrees: bool = False
) -> Any:
    '''利用haversine公式计算两点间的圆心角.'''
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


def get_ellipse(
    x: float = 0,
    y: float = 0,
    a: float = 1,
    b: Optional[float] = None,
    angle: float = 0,
    npts: int = 100,
    ccw: bool = True,
) -> np.ndarray:
    '''
    生成椭圆的xy坐标序列.

    Parameters
    ----------
    x, y : float, optional
        中心的横纵坐标. 默认为(0, 0).

    a : float, optional
        半长轴长度. 默认为1

    b : float, optional
        半短轴长度. 默认为None, 表示和a相等.

    angle : float, optional
        半长轴和x轴成的角度. 默认为0度.

    npts : int, optional
        用linspace(0, 2pi, npts)生成npts个角度.
        默认为100, 要求大于等于4

    ccw : bool, optional
        坐标序列是否沿逆时针方向. 默认为True.

    Returns
    -------
    verts : (npts, 2) np.ndarray
        xy坐标序列.
    '''
    if npts < 4:
        raise ValueError('npts < 4')
    t = np.linspace(0, 2 * np.pi, npts)
    verts = np.c_[np.cos(t), np.sin(t), np.ones_like(t)]

    # 对单位圆做仿射变换.
    b = a if b is None else b
    angle = math.radians(angle)
    cos = math.cos(angle)
    sin = math.sin(angle)
    mtx = [[a * cos, a * sin], [-b * sin, b * cos], [x, y]]
    verts = verts @ mtx
    if not ccw:
        verts = np.flipud(verts)

    return verts


def get_circle(
    x: float = 0, y: float = 0, r: float = 1, npts: int = 100, ccw: bool = True
) -> np.ndarray:
    '''
    生成圆的xy坐标序列.

    Parameters
    ----------
    x, y : float, optional
        中心的横纵坐标. 默认为(0, 0).

    r : float, optional
        圆的半径. 默认为1

    npts : int, optional
        用linspace(0, 2pi, npts)生成npts个角度.
        默认为100, 要求大于等于4

    ccw : bool, optional
        坐标序列是否沿逆时针方向. 默认为True.

    Returns
    -------
    verts : (npts, 2) np.ndarray
        xy坐标序列.
    '''
    return get_ellipse(x, y, r, npts=npts, ccw=ccw)


def region_ind(
    lon: Any, lat: Any, extents: Any, form: Literal['mask', 'ix'] = 'mask'
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    '''
    返回落入给定经纬度方框范围内的索引.

    Parameters
    ----------
    lon : array_like
        经度. 若form='mask'则要求形状与lat一致.

    lat : array_like
        纬度. 若form='mask'则要求形状与lon一致.

    extents : (4,) array_like
        经纬度方框的范围[lon0, lon1, lat0, lat1].

    form : {'mask', 'ix'}
        索引的形式.
        'mask': 使用与lon和lat同形状的布尔数组进行索引.
        'ix': 使用下标构成的开放网格进行索引.

    Returns
    -------
    ind : ndarray or 2-tuple of ndarray
        索引的布尔数组, 或者两个下标数组构成的元组.

    Examples
    --------
    mask = region_ind(lon2d, lat2d, extents)
    data1d = data2d[mask]

    ixgrid = region_ind(lon1d, lat1d, extents, form='ix')
    data2d_subset = data2d[ixgrid]
    data3d_subset = data3d[:, ixgrid[0], ixgrid[1]]
    '''
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    lon0, lon1, lat0, lat1 = extents
    lon_mask = (lon >= lon0) & (lon <= lon1)
    lat_mask = (lat >= lat0) & (lat <= lat1)
    if form == 'mask':
        if lon.shape != lat.shape:
            raise ValueError('lon和lat的形状应该一样')
        ind = lon_mask & lat_mask
    elif form == 'ix':
        ind = np.ix_(lon_mask, lat_mask)
    else:
        raise ValueError('form不支持')

    return ind


def interp_nearest_dd(
    points: Any,
    values: Any,
    xi: Any,
    radius: float = np.inf,
    fill_value: float = np.nan,
) -> np.ndarray:
    '''
    可以限制搜索半径的多维最近邻插值.

    Parameters
    ----------
    points : (n, d) array_like
        n个数据点的坐标, 每个点有d个坐标分量.

    values : (n, ...) array_like
        n个数据点的变量值, 可以有多个波段.

    xi: (m, d) array_like
        m个插值点的坐标. 每个点有d个坐标分量.

    radius : float, optional
        插值点能匹配到数据点的最大距离(半径). 默认为Inf.

    fill_value : float, optional
        距离超出radius的插值点用fill_value填充. 默认为NaN.

    Returns
    -------
    result : (m, ...) ndarray
        插值点处变量值的数组(浮点型).
    '''
    points = np.asarray(points)
    values = np.asarray(values, dtype=float)
    xi = np.asarray(xi)
    if points.ndim != 2:
        raise ValueError('points的维度应该为2')
    if xi.ndim != 2:
        raise ValueError('xi的维度应该为2')
    if values.shape[0] != points.shape[0]:
        raise ValueError('values和points的形状不匹配')

    from scipy.spatial import KDTree

    # 利用KDTree搜索最近点.
    tree = KDTree(points)
    dd, ii = tree.query(xi)
    result = values[ii]
    result[dd > radius] = fill_value

    return result


def interp_nearest_2d(
    x: Any,
    y: Any,
    values: Any,
    xi: Any,
    yi: Any,
    radius: float = np.inf,
    fill_value: float = np.nan,
) -> np.ndarray:
    '''
    可以限制搜索半径的二维最近邻插值.

    Parameters
    ----------
    x : array_like
        数据点的横坐标. 要求形状与y相同.

    y : array_like
        数据点的纵坐标. 要求形状与x相同.

    values : array_like
        数据点的变量值, 可以有多个波段. 要求前面维度的形状与x相同.

    xi: array_like
        插值点的横坐标. 要求形状与yi相同.

    yi: array_like
        插值点的纵坐标. 要求形状与xi相同.

    radius : float, optional
        插值点能匹配到数据点的最大距离(半径). 默认为Inf.

    fill_value : float, optional
        距离超出radius的插值点用fill_value填充. 默认为NaN.

    Returns
    -------
    result : ndarray
        插值点处变量值的数组(浮点型).
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)
    xi = np.asarray(xi)
    yi = np.asarray(yi)

    if x.shape != y.shape:
        raise ValueError('x和y的形状应该一样')
    if values.shape[: x.ndim] != x.shape:
        raise ValueError('values和x的形状不匹配')
    if xi.shape != yi.shape:
        raise ValueError('xi和yi的形状应该一样')

    # 元组解包能避免出现多余的维度.
    band_shape = values.shape[x.ndim :]
    return interp_nearest_dd(
        points=np.c_[x.ravel(), y.ravel()],
        values=values.reshape(-1, *band_shape),
        xi=np.c_[xi.ravel(), yi.ravel()],
        radius=radius,
        fill_value=fill_value,
    ).reshape(*xi.shape, *band_shape)


def binned_average_2d(
    x: Any, y: Any, values: Any, xbins: Any, ybins: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    用平均的方式对数据做二维binning.

    Parameters
    ----------
    x : (n,) array_like
        数据点的横坐标.

    y : (n,) array_like
        数据点的纵坐标.

    values : (n,) array_like or (m,) sequence of array_like
        数据点的变量值, 也可以是一组变量.

    xbins : (nx + 1,) array_like
        用于划分横坐标的bins.

    ybins : (ny + 1,) array_like
        用于划分纵坐标的bins.

    Returns
    -------
    xc : (nx,) ndarray
        xbins每个bin中心的横坐标.

    yc : (ny,) ndarray
        ybins每个bin中心的纵坐标.

    avg : (ny, nx) or (m, ny, nx) ndarray
        xbins和ybins构成的网格内的变量平均值.
        为了便于画图, 采取(ny, nx)的形状.
    '''

    def nanmean(arr):
        '''避免空切片警告的nanmean.'''
        arr = arr[~np.isnan(arr)]
        return np.nan if arr.size == 0 else arr.mean()

    from scipy.stats import binned_statistic_2d

    # 参数检查由scipy的函数负责. 注意x和y的顺序.
    avg, ybins, xbins, _ = binned_statistic_2d(
        y, x, values, bins=[ybins, xbins], statistic=nanmean
    )
    xc = (xbins[1:] + xbins[:-1]) / 2
    yc = (ybins[1:] + ybins[:-1]) / 2

    return xc, yc, avg
