import numpy as np
from scipy.spatial import KDTree
from scipy.stats import binned_statistic_2d

def lon_to_180(lon):
    '''将经度换算到[-180, 180]范围内. 注意180会变成-180.'''
    return (lon + 180) % 360 - 180

def lon_to_360(lon):
    '''将经度换算到[0, 360]范围内.'''
    return lon % 360

def month_to_season(month):
    '''将月份换算为季节. 月份用[1, 12]表示, 季节用[1, 4]表示.'''
    return month % 12 // 3 + 1

def polar_to_xy(r, phi, radians=True):
    '''极坐标转直角坐标.'''
    if not radians:
        phi = np.deg2rad(phi)
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y

def xy_to_polar(x, y, radians=True):
    '''直角坐标转极坐标. 角度范围[-pi, pi].'''
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    if not radians:
        phi = np.rad2deg(phi)

    return r, phi

def wswd_to_uv(ws, wd):
    '''风向风速转为uv.'''
    wd = np.deg2rad(wd)
    u = -ws * np.sin(wd)
    v = -ws * np.cos(wd)

    return u, v

def uv_to_wswd(u, v):
    '''uv转为风向风速.'''
    ws = np.hypot(u, v)
    wd = np.rad2deg(np.arctan2(u, v)) + 180

    return ws, wd

def region_ind(lon, lat, extents, form='mask'):
    '''
    返回落入给定经纬度方框范围内的索引.

    Parameters
    ----------
    lon : ndarray
        经度数组. 若form='mask'则要求形状与lat一致.

    lat : ndarray
        纬度数组. 若form='mask'则要求形状与lon一致.

    extents : 4-tuple of float
        经纬度方框的范围[lonmin, lonmax, latmin, latmax].

    form : {'mask', 'ix'}
        索引的形式.
        'mask': 使用与lon和lat同形状的布尔数组进行索引.
        'ix': 使用下标构成的开放网格进行索引.

    Returns
    -------
    ind : ndarray or 2-tuple of ndarray
        索引的布尔数组或下标数组构成的元组.

    Examples
    --------
    mask = region_ind(lon2d, lat2d, extents)
    data1d = data2d[mask]

    ixgrid = region_ind(lon1d, lat1d, extents, form='ix')
    data2d_subset = data2d[ixgrid]
    data3d_subset = data3d[:, ixgrid[0], ixgrid[1]]
    '''
    lonmin, lonmax, latmin, latmax = extents
    mask_lon = (lon >= lonmin) & (lon <= lonmax)
    mask_lat = (lat <= latmin) & (lat <= latmax)
    if form == 'mask':
        if lon.shape != lat.shape:
            raise ValueError('lon和lat的形状不匹配.')
        ind = mask_lon & mask_lat
    elif form == 'ix':
        ind = np.ix_(mask_lon, mask_lat)
    else:
        raise ValueError('form不支持')

    return ind

def interp_nearest(points, values, xi, radius, fill_value=np.nan):
    '''
    最邻近插值. 接口仿照griddata.

    Parameters
    ----------
    points : (n, D) ndarray or D-tuple of (n,) ndarrays
        n个数据点的坐标. 每个点有D维个坐标分量.

    values : (n,) ndarray
        n个数据点的值.

    xi : (m, D) ndarray or D-tuple of (m,) ndarrays
        m个插值点的坐标. 每个点有D维个坐标分量.

    radius : float
        插值点能匹配到数据点的最大距离(半径).

    fill_value : float, optional
        距离超出radius的插值点用fill_value填充. 默认为NaN.

    Returns
    -------
    result : (m,) ndarray
        m个插值点的值.
    '''
    if isinstance(points, tuple):
        points = np.column_stack(points)
    if isinstance(xi, tuple):
        xi = np.column_stack(xi)
    values = np.asarray(values)

    # 利用KDTree搜索最邻近的值.
    tree = KDTree(points)
    dist, inds = tree.query(xi)
    result = values[inds]
    result[dist > radius] = fill_value

    return result

def bin_avg(lon, lat, data, bins_lon, bins_lat):
    '''
    利用平均的方式对散点数据进行二维binning.

    Parameters
    ----------
    lon : (npt,) array_like
        一维经度数组.

    lat : (npt,) array_like
        一维纬度数组.

    data : (npt,) array_like or (nvar,) list of (npt,) array_like
        一维变量数组. 如果是多个变量组成的列表, 那么分别对每个变量进行计算.

    bins_lon : (nlon + 1,) array_like
        用于划分经度的bins.

    bins_lat : (nlat + 1,) array_like
        用于划分纬度的bins.

    Returns
    -------
    glon : (nlon,) ndarray
        网格经度.

    glat : (nlat,) ndarray
        网格纬度.

    avg : (nlat, nlon) or (nvar, nlat, nlon) ndarray
        每个网格点的平均值. 为了便于画图, 纬度维在前.
    '''
    def func(arr):
        '''避免空切片警告的nanmean.'''
        arr = arr[~np.isnan(arr)]
        return np.nan if arr.size == 0 else arr.mean()

    avg, bins_lat, bins_lon, _ = binned_statistic_2d(
        lat, lon, data, bins=[bins_lat, bins_lon],
        statistic=func
    )
    glon = (bins_lon[1:] + bins_lon[:-1]) / 2
    glat = (bins_lat[1:] + bins_lat[:-1]) / 2

    return glon, glat, avg