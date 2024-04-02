import math
from collections.abc import Sequence, Callable
from typing import Any, Union, Optional

import numpy as np
import pandas as pd

import shapely.geometry as sgeom
from shapely.geometry.base import BaseGeometry, CoordinateSequence
from shapely.prepared import prep
from pyproj import Transformer

from matplotlib.path import Path
from cartopy.crs import CRS

from frykit import DATA_DIRPATH
from frykit._shp import BinaryReader, PolygonType

'''
数据源:

- 中国国界, 省界和市界: https://lbs.amap.com/api/webservice/guide/api/district
- 九段线: https://datav.aliyun.com/portal/school/atlas/area_selector
- 所有国家: http://meteothink.org/downloads/index.html
- 海陆: https://www.naturalearthdata.com/downloads/50m-physical-vectors/
'''

GetCNKeyword = Union[str, Sequence[str]]
GetCNResult = Union[PolygonType, list[PolygonType]]

# 省界和市界的表格.
shp_dirpath = DATA_DIRPATH / 'shp'
pr_filepath = shp_dirpath / 'cn_province.csv'
ct_filepath = shp_dirpath / 'cn_city.csv'
PR_TABLE = pd.read_csv(str(pr_filepath), index_col='pr_name')
CT_TABLE = pd.read_csv(str(ct_filepath), index_col=['pr_name', 'ct_name'])


def get_cn_border() -> PolygonType:
    '''获取中国国界的多边形.'''
    with BinaryReader(shp_dirpath / 'cn_border.bin') as reader:
        return reader.shape(0)


def get_nine_line() -> PolygonType:
    '''获取九段线的多边形.'''
    with BinaryReader(shp_dirpath / 'nine_line.bin') as reader:
        return reader.shape(0)


def _get_pr_locs(province: Optional[GetCNKeyword] = None) -> list[int]:
    '''查询PR_TABLE得到整数索引.'''
    if province is None:
        return list(range(PR_TABLE.shape[0]))
    provinces = [province] if isinstance(province, str) else province
    locs = list(map(PR_TABLE.index.get_loc, provinces))

    return locs


def _get_ct_locs(
    city: Optional[GetCNKeyword] = None, province: Optional[GetCNKeyword] = None
) -> list[int]:
    '''查询CT_TABLE得到整数索引.'''
    if city is not None and province is not None:
        raise ValueError('不能同时指定city和province')
    if city is None:
        city = slice(None)
    if province is None:
        province = slice(None)
    locs = CT_TABLE.index.get_locs([province, city]).tolist()

    return locs


def get_cn_province(province: Optional[GetCNKeyword] = None) -> GetCNResult:
    '''
    获取中国省界的多边形.

    Parameters
    ----------
    province : GetCNKeyword, optional
        单个省名或一组省名. 默认为None, 表示获取所有省.

    Returns
    -------
    result : GetCNResult
        表示省界的多边形. province不是字符串时返回列表.
    '''
    with BinaryReader(shp_dirpath / 'cn_province.bin') as reader:
        result = [reader.shape(i) for i in _get_pr_locs(province)]
    if isinstance(province, str):
        result = result[0]

    return result


def get_cn_city(
    city: Optional[GetCNKeyword] = None, province: Optional[GetCNKeyword] = None
) -> GetCNResult:
    '''
    获取中国市界的多边形.

    Parameters
    ----------
    city : GetCNKeyword, optional
        单个市名或一组市名. 默认为None, 表示获取所有市.

    province : GetCNKeyword, optional
        单个省名或一组省名, 获取属于某个省的所有市.
        默认为None, 表示不使用省名获取.
        不能同时指定city和province.

    Returns
    -------
    result : GetCNResult
        表示市界的多边形. city不是字符串时返回列表.
    '''
    with BinaryReader(shp_dirpath / 'cn_city.bin') as reader:
        result = [reader.shape(i) for i in _get_ct_locs(city, province)]
    if isinstance(city, str):
        result = result[0]

    return result


def get_cn_province_names(short=False) -> list[str]:
    '''获取所有中国省名.'''
    if short:
        names = PR_TABLE['short_name']
    else:
        names = PR_TABLE.index

    return names.to_list()


def get_cn_city_names(short=False) -> list[str]:
    '''获取所有中国市名.'''
    if short:
        names = CT_TABLE['short_name']
    else:
        names = CT_TABLE.index.get_level_values(1)

    return names.to_list()


def get_cn_province_lonlats() -> np.ndarray:
    '''获取所有中国省的经纬度.'''
    return PR_TABLE[['lon', 'lat']].to_numpy()


def get_cn_city_lonlats() -> np.ndarray:
    '''获取所有中国市的经纬度.'''
    return CT_TABLE[['lon', 'lat']].to_numpy()


def get_countries() -> list[PolygonType]:
    '''获取所有国家国界的多边形.'''
    with BinaryReader(shp_dirpath / 'country.bin') as reader:
        return reader.shapes()


def get_land() -> PolygonType:
    '''获取陆地多边形.'''
    with BinaryReader(shp_dirpath / 'land.bin') as reader:
        return reader.shape(0)


def get_ocean() -> PolygonType:
    '''获取海洋多边形.'''
    with BinaryReader(shp_dirpath / 'ocean.bin') as reader:
        return reader.shape(0)


def _ring_codes(n: int) -> list[np.uint8]:
    '''为长度为n的环生成codes.'''
    codes = [Path.LINETO] * n
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    return codes


def polygon_to_path(polygon: PolygonType, keep_empty: bool = True) -> Path:
    '''
    将多边形转为Path.

    Parameters
    ----------
    polygon : PolygonType
        多边形对象.

    keep_empty : bool, optional
        是否用只含(0, 0)点的Path表示空多边形. 默认为True.
        这样在占位的同时不会影响Matplotlib的画图效果.

    Returns
    -------
    path : Path
        Path对象.
    '''
    if not isinstance(polygon, (sgeom.Polygon, sgeom.MultiPolygon)):
        raise TypeError('polygon不是多边形对象')

    if polygon.is_empty:
        if keep_empty:
            return Path([(0, 0)])
        else:
            raise ValueError('polygon不能为空多边形')

    # 注意: 用户构造的多边形不一定满足内外环绕行方向相反的前提.
    vertices, codes = [], []
    for polygon in getattr(polygon, 'geoms', [polygon]):
        for ring in [polygon.exterior, *polygon.interiors]:
            vertices.append(np.asarray(ring.coords))
            codes.extend(_ring_codes(len(ring.coords)))
    vertices = np.vstack(vertices)
    path = Path(vertices, codes)

    return path


def polygon_to_polys(polygon: PolygonType) -> list[list[tuple[float, float]]]:
    '''多边形对象转为适用于shapefile的坐标序列. 但不保证绕行方向.'''
    polys = []
    for polygon in getattr(polygon, 'geoms', [polygon]):
        for ring in [polygon.exterior, *polygon.interiors]:
            polys.append(ring.coords[:])

    return polys


def polygon_to_mask(polygon: PolygonType, x: Any, y: Any) -> np.ndarray:
    '''
    用多边形制作掩膜(mask)数组.

    Parameters
    ----------
    polygon : PolygonType
        多边形对象.

    x : array_like
        数据点的横坐标. 要求形状与y相同.

    y : array_like
        数据点的纵坐标. 要求形状与x相同.

    Returns
    -------
    mask : ndarray
        布尔类型的掩膜数组, 真值表示数据点落入多边形内部. 形状与x和y相同.
    '''
    if not isinstance(polygon, (sgeom.Polygon, sgeom.MultiPolygon)):
        raise TypeError('polygon不是多边形对象')
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        raise ValueError('x和y的形状不匹配')
    if x.ndim == 0 and y.ndim == 0:
        return polygon.contains(sgeom.Point(x, y))
    prepared = prep(polygon)

    def recursion(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''递归判断坐标为x和y的点集是否落入多边形中.'''
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xflag = math.isclose(xmin, xmax)
        yflag = math.isclose(ymin, ymax)
        mask = np.zeros(x.shape, dtype=bool)

        # 散点重合为单点的情况.
        if xflag and yflag:
            point = sgeom.Point(xmin, ymin)
            if prepared.contains(point):
                mask[:] = True
            else:
                mask[:] = False
            return mask

        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        # 散点落在水平和垂直直线上的情况.
        if xflag or yflag:
            line = sgeom.LineString([(xmin, ymin), (xmax, ymax)])
            if prepared.contains(line):
                mask[:] = True
            elif prepared.intersects(line):
                if xflag:
                    m1 = (y >= ymin) & (y <= ymid)
                    m2 = (y >= ymid) & (y <= ymax)
                if yflag:
                    m1 = (x >= xmin) & (x <= xmid)
                    m2 = (x >= xmid) & (x <= xmax)
                if m1.any():
                    mask[m1] = recursion(x[m1], y[m1])
                if m2.any():
                    mask[m2] = recursion(x[m2], y[m2])
            else:
                mask[:] = False
            return mask

        # 散点可以张成矩形的情况.
        box = sgeom.box(xmin, ymin, xmax, ymax)
        if prepared.contains(box):
            mask[:] = True
        elif prepared.intersects(box):
            m1 = (x >= xmid) & (x <= xmax) & (y >= ymid) & (y <= ymax)
            m2 = (x >= xmin) & (x <= xmid) & (y >= ymid) & (y <= ymax)
            m3 = (x >= xmin) & (x <= xmid) & (y >= ymin) & (y <= ymid)
            m4 = (x >= xmid) & (x <= xmax) & (y >= ymin) & (y <= ymid)
            if m1.any():
                mask[m1] = recursion(x[m1], y[m1])
            if m2.any():
                mask[m2] = recursion(x[m2], y[m2])
            if m3.any():
                mask[m3] = recursion(x[m3], y[m3])
            if m4.any():
                mask[m4] = recursion(x[m4], y[m4])
        else:
            mask[:] = False

        return mask

    return recursion(x, y)


def _transform(
    func: Callable[[CoordinateSequence], np.ndarray], geom: BaseGeometry
) -> BaseGeometry:
    '''shapely.ops.transform的修改版, 会将坐标含无效值的对象设为空对象.'''
    if geom.is_empty:
        return type(geom)()
    if geom.geom_type in ('Point', 'LineString', 'LinearRing'):
        coords = func(geom.coords)
        if np.isfinite(coords).all():
            return type(geom)(coords)
        else:
            return type(geom)()
    elif geom.geom_type == 'Polygon':
        shell = func(geom.exterior.coords)
        if not np.isfinite(shell).all():
            return sgeom.Polygon()
        holes = []
        for ring in geom.interiors:
            hole = func(ring.coords)
            if np.isfinite(hole).all():
                holes.append(hole)
        return sgeom.Polygon(shell, holes)
    elif (
        geom.geom_type.startswith('Multi')
        or geom.geom_type == 'GeometryCollection'
    ):
        parts = []
        for part in geom.geoms:
            part = _transform(func, part)
            if not part.is_empty:
                parts.append(part)
        if parts:
            return type(geom)(parts)
        else:
            return type(geom)()
    else:
        raise TypeError('geom不是几何对象')


class GeometryTransformer:
    '''
    对几何对象做坐标变换的类.

    基于pyproj.Transformer实现, 当几何对象跨越坐标系边界时可能产生错误的连线.
    '''

    def __init__(self, crs_from: CRS, crs_to: CRS) -> None:
        '''
        Parameters
        ----------
        crs_from : CRS
            源坐标系.

        crs_to : CRS
            目标坐标系.
        '''
        self.crs_from = crs_from
        self.crs_to = crs_to

        # 坐标系相同时直接复制.
        if crs_from == crs_to:
            self._func = lambda x: type(x)(x)
            return None

        # 避免反复创建Transformer的开销.
        transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)

        def func(coords: CoordinateSequence) -> np.ndarray:
            coords = np.asarray(coords)
            return np.column_stack(
                transformer.transform(coords[:, 0], coords[:, 1])
            ).squeeze()

        self._func = lambda x: _transform(func, x)

    def __call__(self, geom: BaseGeometry) -> BaseGeometry:
        '''
        对几何对象做变换.

        Parameters
        ----------
        geom : BaseGeometry
            源坐标系上的几何对象.

        Returns
        -------
        geom : BaseGeometry
            目标坐标系上的几何对象.
        '''
        return self._func(geom)
