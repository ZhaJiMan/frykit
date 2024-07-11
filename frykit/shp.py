import math
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, Optional, Union

import numpy as np
import pandas as pd  # 加载 overhead 还挺高
import shapely.geometry as sgeom
from cartopy.crs import CRS
from matplotlib.path import Path
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry, CoordinateSequence
from shapely.prepared import prep

from frykit import SHP_DIRPATH
from frykit._shp import BinaryReader
from frykit.help import deprecator

'''
数据源

- 省市县: https://lbs.amap.com/api/webservice/guide/api/district
- 九段线: https://datav.aliyun.com/portal/school/atlas/area_selector
- 所有国家: http://meteothink.org/downloads/index.html
- 海陆: https://www.naturalearthdata.com/downloads/50m-physical-vectors/
'''

StrOrInt = Union[str, int]
DictOrList = Union[dict, list[dict]]
PolygonType = Union[sgeom.Polygon, sgeom.MultiPolygon]
PolygonOrList = Union[PolygonType, list[PolygonType]]

GetCnKey = Union[StrOrInt, Sequence[StrOrInt]]
GetCnResult = Union[PolygonOrList, DictOrList]


# 缓存多边形数据
_data_cache = {}


def get_cn_border() -> sgeom.MultiPolygon:
    '''获取中国国界的多边形'''
    polygon = _data_cache.get('cn_border')
    if polygon is None:
        filepath = SHP_DIRPATH / 'cn_border.bin'
        with BinaryReader(filepath, region='china') as reader:
            polygon = reader.shape(0)
        _data_cache['cn_border'] = polygon

    return polygon


def get_nine_line() -> sgeom.MultiPolygon:
    '''获取九段线的多边形'''
    polygon = _data_cache.get('nine_line')
    if polygon is None:
        filepath = SHP_DIRPATH / 'nine_line.bin'
        with BinaryReader(filepath, region='china') as reader:
            polygon = reader.shape(0)
        _data_cache['nine_line'] = polygon

    return polygon


def get_cn_province_table() -> pd.DataFrame:
    '''获取省界元数据的表格'''
    filepath = SHP_DIRPATH / 'cn_province.csv'
    return pd.read_csv(str(filepath))


def get_cn_city_table() -> pd.DataFrame:
    '''获取市界元数据的表格'''
    filepath = SHP_DIRPATH / 'cn_city.csv'
    return pd.read_csv(str(filepath))


def get_cn_district_table() -> pd.DataFrame:
    '''获取县界元数据的表格'''
    filepath = SHP_DIRPATH / 'cn_district.csv'
    return pd.read_csv(str(filepath))


_PR_TABLE = None
_CT_TABLE = None
_DT_TABLE = None


def _get_pr_table() -> pd.DataFrame:
    '''获取缓存的省界元数据的表格'''
    global _PR_TABLE
    if _PR_TABLE is None:
        _PR_TABLE = get_cn_province_table()

    return _PR_TABLE


def _get_ct_table() -> pd.DataFrame:
    '''获取缓存的市界元数据的表格'''
    global _CT_TABLE
    if _CT_TABLE is None:
        _CT_TABLE = get_cn_city_table()

    return _CT_TABLE


def _get_dt_table() -> pd.DataFrame:
    '''获取缓存的县界元数据的表格'''
    global _DT_TABLE
    if _DT_TABLE is None:
        _DT_TABLE = get_cn_district_table()

    return _DT_TABLE


def _get_locs(index: pd.Index, key: Any) -> list[int]:
    '''保证返回整数下标列表的 Index.get_loc'''
    loc = index.get_loc(key)
    if isinstance(loc, slice):
        return list(range(len(index))[loc])
    if isinstance(loc, np.ndarray):
        return np.nonzero(loc)[0].tolist()

    return [loc]


def _get_pr_locs(province: Optional[GetCnKey] = None) -> list[int]:
    '''查询省界元数据表格的下标'''
    df = _get_pr_table()
    if province is None:
        return list(range(len(df)))

    names = pd.Index(df['pr_name'])
    adcodes = pd.Index(df['pr_adcode'])

    def func(key: GetCnKey) -> list[int]:
        if isinstance(key, str):
            return _get_locs(names, key)
        if isinstance(key, int):
            return _get_locs(adcodes, key)
        return list(chain(*map(func, key)))

    return func(province)


def _get_ct_locs(
    city: Optional[GetCnKey] = None, province: Optional[GetCnKey] = None
) -> list[int]:
    '''查询市界元数据表格的下标'''
    df = _get_ct_table()
    if city is None and province is None:
        return list(range(len(df)))
    if city is not None and province is not None:
        raise ValueError('不能同时指定 city 和 province')

    if city is not None:
        names = pd.Index(df['ct_name'])
        adcodes = pd.Index(df['ct_adcode'])
        key = city

    if province is not None:
        names = pd.Index(df['pr_name'])
        adcodes = pd.Index(df['pr_adcode'])
        key = province

    def func(key: GetCnKey) -> list[int]:
        if isinstance(key, str):
            return _get_locs(names, key)
        if isinstance(key, int):
            return _get_locs(adcodes, key)
        return list(chain(*map(func, key)))

    return func(key)


def _get_dt_locs(
    district: Optional[GetCnKey] = None,
    city: Optional[GetCnKey] = None,
    province: Optional[GetCnKey] = None,
) -> list[int]:
    '''查询县界元数据表格的下标'''
    df = _get_dt_table()
    num_keys = sum(key is not None for key in [district, city, province])
    if num_keys == 0:
        return list(range(len(df)))
    if num_keys >= 2:
        raise ValueError('district、city 和 province 三个参数中只能指定一个')

    if district is not None:
        names = pd.Index(df['dt_name'])
        adcodes = pd.Index(df['dt_adcode'])
        key = district

    if city is not None:
        names = pd.Index(df['ct_name'])
        adcodes = pd.Index(df['ct_adcode'])
        key = city

    if province is not None:
        names = pd.Index(df['pr_name'])
        adcodes = pd.Index(df['pr_adcode'])
        key = province

    def func(key: GetCnKey) -> list[int]:
        if isinstance(key, str):
            return _get_locs(names, key)
        if isinstance(key, int):
            return _get_locs(adcodes, key)
        return list(chain(*map(func, key)))

    locs = func(key)
    if isinstance(district, str) and len(locs) > 1:
        lines = []
        for row in df.iloc[locs].itertuples(index=False):
            parts = [
                f'province={row.pr_name}',
                f'city={row.ct_name}',
                f'district={row.dt_name}',
                f'adcode={row.dt_adcode}',
            ]
            line = ', '.join(parts)
            lines.append(line)

        lines = '\n'.join(lines)
        msg = f'存在复数个同名的区县，请用 adcode 指定\n{lines}'
        raise ValueError(msg)

    return locs


def get_cn_province_names(short: bool = False) -> list[str]:
    '''获取中国省名'''
    df = _get_pr_table()
    key = 'short_name' if short else 'pr_name'
    names = df[key].tolist()

    return names


def get_cn_city_names(
    province: Optional[GetCnKey] = None, short=False
) -> list[str]:
    '''获取中国市名。可以指定获取某个省的所有市名。'''
    df = _get_ct_table()
    locs = _get_ct_locs(province=province)
    key = 'short_name' if short else 'ct_name'
    names = df[key].iloc[locs].tolist()

    return names


def get_cn_district_names(
    city: Optional[GetCnKey] = None,
    province: Optional[GetCnKey] = None,
    short: bool = False,
) -> list[str]:
    '''获取中国县名。可以指定获取某个市或某个省的所有县名。'''
    df = _get_dt_table()
    locs = _get_dt_locs(city=city, province=province)
    key = 'short_name' if short else 'dt_name'
    names = df[key].iloc[locs].tolist()

    return names


def get_cn_province(
    province: Optional[GetCnKey] = None, as_dict: bool = False
) -> GetCnResult:
    '''
    获取中国省界的多边形

    Parameters
    ----------
    province : GetCnKey, optional
        省名或 adcode。可以是复数个省。
        默认为 None，表示获取所有省。

    as_dict : bool, optional
        是否返回带元数据的字典。默认为 False。

    Returns
    -------
    result : GetCnResult
        表示省界的多边形或字典
    '''
    polygons = _data_cache.get('cn_province')
    if polygons is None:
        filepath = SHP_DIRPATH / 'cn_province.bin'
        with BinaryReader(filepath, region='china') as reader:
            polygons = reader.shapes()
        _data_cache['cn_province'] = polygons

    locs = _get_pr_locs(province)
    polygons = [polygons[i] for i in locs]

    if as_dict:
        df = _get_pr_table().iloc[locs]
        result = df.to_dict(orient='records')
        for d, polygon in zip(result, polygons):
            d['geometry'] = polygon
    else:
        result = polygons

    if isinstance(province, (str, int)):
        result = result[0]

    return result


def get_cn_city(
    city: Optional[GetCnKey] = None,
    province: Optional[GetCnKey] = None,
    as_dict: bool = False,
) -> GetCnResult:
    '''
    获取中国市界的多边形

    Parameters
    ----------
    city : GetCnKey, optional
        市名或 adcode。可以是复数个市。
        默认为 None，表示获取所有市。

    province : GetCnKey, optional
        省名或 adcode，表示获取某个省的所有市。可以是复数个省。
        默认为 None，表示不指定省。

    as_dict : bool, optional
        是否返回带元数据的字典。默认为 False。

    Returns
    -------
    result : GetCnResult
        表示市界的多边形或字典
    '''
    polygons = _data_cache.get('cn_city')
    if polygons is None:
        filepath = SHP_DIRPATH / 'cn_city.bin'
        with BinaryReader(filepath, region='china') as reader:
            polygons = reader.shapes()
        _data_cache['cn_city'] = polygons

    locs = _get_ct_locs(city, province)
    polygons = [polygons[i] for i in locs]

    if as_dict:
        df = _get_ct_table().iloc[locs]
        result = df.to_dict(orient='records')
        for d, polygon in zip(result, polygons):
            d['geometry'] = polygon
    else:
        result = polygons

    if isinstance(city, (str, int)):
        result = result[0]

    return result


def get_cn_district(
    district: Optional[GetCnKey] = None,
    city: Optional[GetCnKey] = None,
    province: Optional[GetCnKey] = None,
    as_dict: bool = False,
) -> GetCnResult:
    '''
    获取中国县界的多边形

    Parameters
    ----------
    district : GetCnKey, optional
        县名或 adcode。可以是复数个县。
        默认为 None，表示获取所有县。

    city : GetCnKey, optional
        市名或 adcode，表示获取某个市的所有县。可以是复数个市。
        默认为 None，表示不指定市。

    province : GetCnKey, optional
        省名或 adcode，表示获取某个省的所有县。可以是复数个省。
        默认为 None，表示不指定省。

    as_dict : bool, optional
        是否返回带元数据的字典。默认为 False。

    Returns
    -------
    result : GetCnResult
        表示县界的多边形或字典
    '''
    polygons = _data_cache.get('cn_district')
    if polygons is None:
        filepath = SHP_DIRPATH / 'cn_district.bin'
        with BinaryReader(filepath, region='china') as reader:
            polygons = reader.shapes()
        _data_cache['cn_district'] = polygons

    locs = _get_dt_locs(district, city, province)
    polygons = [polygons[i] for i in locs]

    if as_dict:
        df = _get_dt_table().iloc[locs]
        result = df.to_dict(orient='records')
        for d, polygon in zip(result, polygons):
            d['geometry'] = polygon
    else:
        result = polygons

    if isinstance(district, (str, int)):
        result = result[0]

    return result


def get_countries() -> list[PolygonType]:
    '''获取所有国家国界的多边形'''
    polygons = _data_cache.get('country')
    if polygons is None:
        filepath = SHP_DIRPATH / 'country.bin'
        with BinaryReader(filepath, region='world') as reader:
            polygons = reader.shapes()
        _data_cache['country'] = polygons

    return polygons


def get_land() -> sgeom.MultiPolygon:
    '''获取陆地多边形'''
    polygon = _data_cache.get('land')
    if polygon is None:
        filepath = SHP_DIRPATH / 'land.bin'
        with BinaryReader(filepath, region='world') as reader:
            polygon = reader.shape(0)
        _data_cache['land'] = polygon

    return polygon


def get_ocean() -> sgeom.MultiPolygon:
    '''获取海洋多边形'''
    polygon = _data_cache.get('ocean')
    if polygon is None:
        filepath = SHP_DIRPATH / 'ocean.bin'
        with BinaryReader(filepath, region='world') as reader:
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


# 用于占位的 Path，不会被画出。
PLACEHOLDER_PATH = Path(np.zeros((0, 2)), [])


def polygon_to_path(polygon: PolygonType) -> Path:
    '''多边形转为 Path。空多边形对应空 Path。'''
    if not isinstance(polygon, (sgeom.Polygon, sgeom.MultiPolygon)):
        raise TypeError('polygon 不是多边形')

    if polygon.is_empty:
        return PLACEHOLDER_PATH

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
