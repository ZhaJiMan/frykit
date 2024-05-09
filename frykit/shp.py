import math
from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
import pandas as pd  # 加载 overhead 还挺高
import shapely.geometry as sgeom
from cartopy.crs import CRS
from matplotlib.path import Path
from numpy.distutils.misc_util import is_sequence
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry, CoordinateSequence
from shapely.prepared import prep

from frykit import SHP_DIRPATH
from frykit._shp import BinaryReader
from frykit._typing import StrOrSeq
from frykit.help import deprecator

'''
数据源

- 中国行政区划: https://lbs.amap.com/api/webservice/guide/api/district
- 九段线: https://datav.aliyun.com/portal/school/atlas/area_selector
- 所有国家: http://meteothink.org/downloads/index.html
- 海陆: https://www.naturalearthdata.com/downloads/50m-physical-vectors/
'''

PolygonType = Union[sgeom.Polygon, sgeom.MultiPolygon]
PolygonOrList = Union[PolygonType, list[PolygonType]]

# 缓存地理数据
_data_cache = {}


def get_cn_border() -> sgeom.MultiPolygon:
    '''获取中国国界的多边形'''
    polygon = _data_cache.get('cn_border')
    if polygon is None:
        with BinaryReader(SHP_DIRPATH / 'cn_border.bin') as reader:
            polygon = reader.shape(0)
        _data_cache['cn_border'] = polygon

    return polygon


def get_nine_line() -> sgeom.MultiPolygon:
    '''获取九段线的多边形'''
    polygon = _data_cache.get('nine_line')
    if polygon is None:
        with BinaryReader(SHP_DIRPATH / 'nine_line.bin') as reader:
            polygon = reader.shape(0)
        _data_cache['nine_line'] = polygon

    return polygon


def get_cn_province_table() -> pd.DataFrame:
    '''获取省界元数据的表格'''
    filepath = SHP_DIRPATH / 'cn_province.csv'
    return pd.read_csv(str(filepath), index_col='pr_name')


def get_cn_city_table() -> pd.DataFrame:
    '''获取市界元数据的表格'''
    filepath = SHP_DIRPATH / 'cn_city.csv'
    return pd.read_csv(str(filepath), index_col=['pr_name', 'ct_name'])


_PR_TABLE = get_cn_province_table()
_CT_TABLE = get_cn_city_table()


def _get_pr_locs(province: Optional[StrOrSeq] = None) -> list[int]:
    '''查询 _PR_TABLE 得到整数索引'''
    if province is None:
        return list(range(_PR_TABLE.shape[0]))
    elif is_sequence(province):
        return list(map(_PR_TABLE.index.get_loc, province))
    else:
        return [_PR_TABLE.index.get_loc(province)]


def _get_ct_locs(
    city: Optional[StrOrSeq] = None, province: Optional[StrOrSeq] = None
) -> list[int]:
    '''查询 _CT_TABLE 得到整数索引'''
    if city is not None and province is not None:
        raise ValueError('不能同时指定city和province')
    if city is None:
        city = slice(None)
    if province is None:
        province = slice(None)
    locs = _CT_TABLE.index.get_locs([province, city]).tolist()

    return locs


def get_cn_province_names(short: bool = False) -> list[str]:
    '''获取所有中国省名'''
    if short:
        names = _PR_TABLE['short_name']
    else:
        names = _PR_TABLE.index

    return names.tolist()


def get_cn_city_names(
    province: Optional[StrOrSeq] = None, short=False
) -> list[str]:
    '''获取所有中国市名。可以指定获取某个省的所有市名。'''
    locs = _get_ct_locs(province=province)
    table = _CT_TABLE.iloc[locs]
    if short:
        names = table['short_name']
    else:
        names = table.index.get_level_values(1)

    return names.tolist()


def get_cn_province(province: Optional[StrOrSeq] = None) -> PolygonOrList:
    '''
    获取中国省界的多边形

    Parameters
    ----------
    province : StrOrSeq, optional
        单个或一组省名。默认为 None，表示获取所有省。

    Returns
    -------
    polygons : PolygonOrList
        表示省界的多边形。province 是字符串时返回单个多边形，否则返回列表。
    '''
    polygons = _data_cache.get('cn_province')
    if polygons is None:
        with BinaryReader(SHP_DIRPATH / 'cn_province.bin') as reader:
            polygons = reader.shapes()
        _data_cache['cn_province'] = polygons
    polygons = [polygons[i] for i in _get_pr_locs(province)]

    if isinstance(province, str):
        return polygons[0]

    return polygons


def get_cn_city(
    city: Optional[StrOrSeq] = None, province: Optional[StrOrSeq] = None
) -> PolygonOrList:
    '''
    获取中国市界的多边形

    Parameters
    ----------
    city : StrOrSeq, optional
        单个或一组市名。默认为 None，表示获取所有市。

    province : StrOrSeq, optional
        单个或一组省名，获取属于某个省的所有市。
        默认为 None，表示不指定省名。
        不能同时指定 city 和 province。

    Returns
    -------
    polygons : PolygonOrList
        表示市界的多边形。city 是字符串时返回单个多边形，否则返回列表。
    '''
    polygons = _data_cache.get('cn_city')
    if polygons is None:
        with BinaryReader(SHP_DIRPATH / 'cn_city.bin') as reader:
            polygons = reader.shapes()
        _data_cache['cn_city'] = polygons
    polygons = [polygons[i] for i in _get_ct_locs(city, province)]

    if isinstance(city, str):
        return polygons[0]

    return polygons


def get_countries() -> list[PolygonType]:
    '''获取所有国家国界的多边形'''
    polygons = _data_cache.get('country')
    if polygons is None:
        with BinaryReader(SHP_DIRPATH / 'country.bin') as reader:
            polygons = reader.shapes()
        _data_cache['country'] = polygons

    return polygons


def get_land() -> sgeom.MultiPolygon:
    '''获取陆地多边形'''
    polygon = _data_cache.get('land')
    if polygon is None:
        with BinaryReader(SHP_DIRPATH / 'land.bin') as reader:
            polygon = reader.shape(0)
        _data_cache['land'] = polygon

    return polygon


def get_ocean() -> sgeom.MultiPolygon:
    '''获取海洋多边形'''
    polygon = _data_cache.get('ocean')
    if polygon is None:
        with BinaryReader(SHP_DIRPATH / 'ocean.bin') as reader:
            polygon = reader.shape(0)
        _data_cache['ocean'] = polygon

    return polygon


def _poly_codes(n: int) -> list[np.uint8]:
    '''生成适用于多边形的长度为 n 的 codes'''
    codes = [Path.LINETO] * n
    if codes:
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY

    return codes


def polygon_to_path(polygon: PolygonType) -> Path:
    '''多边形转为 Path。空多边形对应空 Path。'''
    if not isinstance(polygon, (sgeom.Polygon, sgeom.MultiPolygon)):
        raise TypeError('polygon 不是多边形')

    if polygon.is_empty:
        return Path(np.zeros((0, 2)), [])

    verts = []
    codes = []
    for polygon in getattr(polygon, 'geoms', [polygon]):
        coords = polygon.exterior.coords
        if polygon.exterior.is_ccw:
            coords = coords[::-1]
        verts.append(coords)
        codes.extend(_poly_codes(len(coords)))

        for ring in polygon.interiors:
            coords = ring.coords
            if not ring.is_ccw:
                coords = coords[::-1]
            verts.append(coords)
            codes.extend(_poly_codes(len(coords)))

    verts = np.vstack(verts)
    path = Path(verts, codes)

    return path


def path_to_polygon(path: Path) -> PolygonType:
    '''Path 转为多边形。注意只适用于 polygon_to_path 的返回值。'''
    if len(path.vertices) == 0:
        return sgeom.Polygon()

    collection = []
    inds = np.nonzero(path.codes == Path.MOVETO)[0][1:]
    for verts in np.vsplit(path.vertices, inds):
        ring = sgeom.LinearRing(verts)
        if ring.is_ccw:
            collection[-1][1].append(ring)
        else:
            collection.append((ring, []))

    polygons = [sgeom.Polygon(shell, holes) for shell, holes in collection]
    if len(polygons) > 1:
        return sgeom.MultiPolygon(polygons)
    return polygons[0]


def polygon_to_polys(polygon: PolygonType) -> list[list[tuple[float, float]]]:
    '''多边形转为适用于 shapefile 的坐标序列。不保证绕行方向。'''
    polys = []
    for polygon in getattr(polygon, 'geoms', [polygon]):
        for ring in [polygon.exterior, *polygon.interiors]:
            polys.append(ring.coords[:])

    return polys


def polygon_to_mask(polygon: PolygonType, x: Any, y: Any) -> np.ndarray:
    '''
    用多边形制作掩膜（mask）数组

    Parameters
    ----------
    polygon : PolygonType
        用于裁剪的多边形

    x : array_like
        数据点的横坐标。要求形状与y相同。

    y : array_like
        数据点的纵坐标。要求形状与x相同。

    Returns
    -------
    mask : ndarray
        布尔类型的掩膜数组，真值表示数据点落入多边形内部。形状与x和y相同。
    '''
    if not isinstance(polygon, (sgeom.Polygon, sgeom.MultiPolygon)):
        raise TypeError('polygon 不是多边形')

    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError('x 和 y 的形状不匹配')
    if x.ndim == 0 and y.ndim == 0:
        return polygon.contains(sgeom.Point(x, y))
    prepared = prep(polygon)

    def recursion(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''递归判断坐标为 x 和 y 的点集是否落入多边形中'''
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xflag = math.isclose(xmin, xmax)
        yflag = math.isclose(ymin, ymax)
        mask = np.zeros(x.shape, dtype=bool)

        # 散点重合为单点的情况
        if xflag and yflag:
            point = sgeom.Point(xmin, ymin)
            if prepared.contains(point):
                mask[:] = True
            else:
                mask[:] = False
            return mask

        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        # 散点落在水平和垂直直线上的情况
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

        # 散点可以张成矩形的情况
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


def _transform(func: Callable, geom: BaseGeometry) -> BaseGeometry:
    '''shapely.ops.transform 的修改版，会将坐标含无效值的对象设为空对象。'''
    if geom.is_empty:
        return type(geom)()
    if geom.geom_type in ('Point', 'LineString', 'LinearRing'):
        coords = func(geom.coords)
        if np.isfinite(coords).all():
            return type(geom)(coords)
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
        return type(geom)()
    else:
        raise TypeError('geom 不是几何对象')


class GeometryTransformer:
    '''
    对几何对象做坐标变换的类

    基于 pyproj.Transformer 实现，可能在地图边界出现错误的结果。
    '''

    def __init__(self, crs_from: CRS, crs_to: CRS) -> None:
        '''
        Parameters
        ----------
        crs_from : CRS
            源坐标系

        crs_to : CRS
            目标坐标系
        '''
        self.crs_from = crs_from
        self.crs_to = crs_to

        # 坐标系相同时直接复制
        if crs_from == crs_to:
            self._func = lambda x: type(x)(x)
            return None

        # 避免反复创建Transformer的开销
        transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)

        def func(coords: CoordinateSequence) -> np.ndarray:
            coords = np.asarray(coords)
            return np.column_stack(
                transformer.transform(coords[:, 0], coords[:, 1])
            ).squeeze()

        self._func = lambda x: _transform(func, x)

    def __call__(self, geom: BaseGeometry) -> BaseGeometry:
        '''
        对几何对象做变换

        Parameters
        ----------
        geom : BaseGeometry
            源坐标系上的几何对象

        Returns
        -------
        geom : BaseGeometry
            目标坐标系上的几何对象
        '''
        return self._func(geom)


@deprecator(get_cn_province, raise_error=True)
def get_cn_shp():
    pass
