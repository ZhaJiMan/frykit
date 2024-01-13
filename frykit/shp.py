import math
from collections.abc import Sequence, Callable
from typing import Any, Union, Optional, Literal

import numpy as np
import pandas as pd

import shapely.geometry as sgeom
from shapely.geometry.base import BaseGeometry, CoordinateSequence
from shapely.prepared import prep
from pyproj import Transformer

from matplotlib.path import Path
from cartopy.crs import CRS

from frykit import DATA_DIRPATH
from frykit._shp import BinaryReader

PolygonType = Union[sgeom.Polygon, sgeom.MultiPolygon]
GetCNResult = Union[PolygonType, dict]

# 省界和市界的表格.
shp_dirpath = DATA_DIRPATH / 'shp'
PROVINCE_TABLE = pd.read_csv(str(shp_dirpath / 'cn_province.csv'))
CITY_TABLE = pd.read_csv(str(shp_dirpath / 'cn_city.csv'))

def get_cn_border(as_dict: bool = False) -> GetCNResult:
    '''
    获取中国国界的多边形.

    Parameters
    ----------
    ad_dict : bool, optional
        是否以字典形式返回结果. 默认为False.
        多边形在字典中的键为'geometry'.

    Returns
    -------
    result : GetCNResult
        表示国界的多边形或字典.
    '''
    with BinaryReader(shp_dirpath / 'cn_border.bin') as reader:
        geom = reader.shape(0)
    if as_dict:
        return {
            'cn_name': '中华人民共和国',
            'cn_adcode': 100000,
            'geometry': geom
        }
    else:
        return geom

def get_nine_line(as_dict: bool = False) -> GetCNResult:
    '''
    获取九段线的多边形.

    Parameters
    ----------
    ad_dict : bool, optional
        是否以字典形式返回结果. 默认为False.
        多边形在字典中的键为'geometry'.

    Returns
    -------
    result : GetCNResult
        表示九段线的多边形或字典.
    '''
    with BinaryReader(shp_dirpath / 'nine_line.bin') as reader:
        geom = reader.shape(0)
    if as_dict:
        return {
            'cn_name': '九段线',
            'cn_adcode': 100000,
            'geometry': geom
        }
    else:
        return geom

def get_cn_province(
    name: Optional[Union[str, Sequence[str]]] = None,
    as_dict: bool = False
) -> Union[GetCNResult, list[GetCNResult]]:
    '''
    获取中国省界的多边形.

    Parameters
    ----------
    name: str or sequence of str, optional
        单个省名或一组省名. 默认为None, 表示获取所有省份.

    ad_dict : bool, optional
        是否以字典形式返回结果. 默认为False.
        多边形在字典中的键为'geometry'.

    Returns
    -------
    result : GetCNResult or list of GetCNResult
        表示省界的多边形或字典.
        当name表示多个省时result是列表.
    '''
    if name is None:
        indexer = slice(None)
    else:
        if isinstance(name, str):
            indexer = PROVINCE_TABLE['pr_name'] == name
        else:
            indexer = PROVINCE_TABLE['pr_name'].isin(name)
        if not indexer.any():
            raise ValueError('name错误')

    result = []
    with BinaryReader(shp_dirpath / 'cn_province.bin') as reader:
        for i, row in PROVINCE_TABLE.loc[indexer].iterrows():
            geom = reader.shape(i)
            if as_dict:
                result.append({**row, 'geometry': geom})
            else:
                result.append(geom)

    # 只含一条结果时返回标量.
    return result if len(result) > 1 else result[0]

def get_cn_city(
    name: Optional[Union[str, Sequence[str]]] = None,
    province: Optional[Union[str, Sequence[str]]] = None,
    as_dict: bool = False
) -> Union[GetCNResult, list[GetCNResult]]:
    '''
    获取中国市界的多边形.

    Parameters
    ----------
    name: str or sequence of str, optional
        单个市名或一组市名. 默认为None, 表示获取所有市.

    province: str or sequence of str, optional
        单个省名或一组省名, 搜索属于某个省的所有市.
        默认为None, 表示不使用省名搜索.
        不能同时指定name和province.

    ad_dict : bool, optional
        是否以字典形式返回结果. 默认为False.
        多边形在字典中的键为'geometry'.

    Returns
    -------
    result : GetCNResult or list of GetCNResult
        表示市界的多边形或字典.
        当name表示多个市或province不为None时result是列表.
    '''
    if name is not None and province is not None:
        raise ValueError('不能同时指定name和province')
    elif name is None and province is None:
        indexer = slice(None)
    elif name is not None:
        if isinstance(name, str):
            indexer = CITY_TABLE['ct_name'] == name
        else:
            indexer = CITY_TABLE['ct_name'].isin(name)
        if not indexer.any():
            raise ValueError('name错误')
    else:
        if isinstance(province, str):
            indexer = CITY_TABLE['pr_name'] == province
        else:
            indexer = CITY_TABLE['pr_name'].isin(province)
        if not indexer.any():
            raise ValueError('province错误')

    result = []
    with BinaryReader(shp_dirpath / 'cn_city.bin') as reader:
        for i, row in CITY_TABLE.loc[indexer].iterrows():
            geom = reader.shape(i)
            if as_dict:
                result.append({**row, 'geometry': geom})
            else:
                result.append(geom)

    # 只含一条结果时返回标量.
    if len(result) > 1 or province is not None:
        return result
    else:
        return result[0]

def get_cn_shp(
    level: Literal['国', '省', '市'] = '国',
    province: Optional[Union[str, Sequence[str]]] = None,
    city: Optional[Union[str, Sequence[str]]] = None,
    as_dict: bool = False
) -> Union[GetCNResult, list[GetCNResult]]:
    '''
    获取中国行政区划的多边形. 支持国界, 省界和市界.

    接口仿照cnmaps.maps.get_adm_maps
    数据源: https://github.com/GaryBikini/ChinaAdminDivisonSHP
    使用了PRCoords库从GCJ-02坐标变换到了WGS-84坐标.

    Examples
    --------
    - 获取国界: get_cn_shp(level='国')
    - 获取所有省: get_cn_shp(level='省')
    - 获取某个省: get_cn_shp(level='省', province='河北省')
    - 获取所有市: get_cn_shp(level='市')
    - 获取某个省的所有市: get_cn_shp(level='市', province='河北省')
    - 获取某个市: get_cn_shp(level='市', city='石家庄市')

    Parameters
    ----------
    level : {'国', '省', '市'}, optional
        边界等级. 默认为'国'.

    province : str or sequence of str, optional
        省名. 可以是字符串或一组字符串, 默认为None.

    city : str or sequence of str, optional
        市名. 可以是字符串或一组字符串, 默认为None.

    as_dict : bool, optional
        是否以字典形式返回结果. 默认为False.
        字典的键大致包括:
        - cn_name, pr_name, ct_name: 国省市名
        - cn_adcode, pr_adcode, ct_adcode: 国省市的区划代码
        - geometry: 多边形对象

    Returns
    -------
    result : GetCNResult or list of GetCNResult
        行政区划的多边形或多边形构成的列表.
        as_dict为True时则是字典或字典构成的列表.
    '''
    if level == '国':
        if province is not None or city is not None:
            raise ValueError("level='国'时不能指定province或city")
        return get_cn_border(as_dict)
    elif level == '省':
        if city is not None:
            raise ValueError("level='省'时不能指定city")
        return get_cn_province(province, as_dict)
    elif level == '市':
        if province is not None and city is not None:
            raise ValueError("level='市'时不能同时指定province和city")
        return get_cn_city(city, province, as_dict)
    else:
        raise ValueError("level只能为{'国', '省', '市'}中的一种")

def get_cn_province_names() -> list[str]:
    '''获取中国省名.'''
    return PROVINCE_TABLE['pr_name'].to_list()

def get_cn_city_names() -> list[str]:
    '''获取中国市名.'''
    return CITY_TABLE['ct_name'].to_list()

def simplify_province_name(name: str) -> str:
    '''简化省名到2或3个字.'''
    if name.startswith('内蒙古') or name.startswith('黑龙江'):
        return name[:3]
    else:
        return name[:2]

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

    # 用多边形含有的所有环的顶点构造Path.
    vertices, codes = [], []
    for polygon in getattr(polygon, 'geoms', [polygon]):
        for ring in [polygon.exterior, *polygon.interiors]:
            vertices.append(np.asarray(ring.coords))
            codes.extend(_ring_codes(len(ring.coords)))
    vertices = np.vstack(vertices)
    path = Path(vertices, codes)

    return path

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

    def recursion(x, y):
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
                if m1.any(): mask[m1] = recursion(x[m1], y[m1])
                if m2.any(): mask[m2] = recursion(x[m2], y[m2])
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
            if m1.any(): mask[m1] = recursion(x[m1], y[m1])
            if m2.any(): mask[m2] = recursion(x[m2], y[m2])
            if m3.any(): mask[m3] = recursion(x[m3], y[m3])
            if m4.any(): mask[m4] = recursion(x[m4], y[m4])
        else:
            mask[:] = False

        return mask

    return recursion(x, y)

def _transform(
    func: Callable[[CoordinateSequence], np.ndarray],
    geom: BaseGeometry
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

def transform_geometries(
    geoms: Sequence[BaseGeometry],
    crs_from: CRS,
    crs_to: CRS
) -> list[BaseGeometry]:
    '''
    将一组几何对象从crs_from坐标系变换到crs_to坐标系上.

    基于pyproj.Transformer实现. 相比cartopy.crs.Projection.project_geometry
    速度更快, 但当几何对象跨越坐标系边界时可能产生错误的连线.

    Parameters
    ----------
    geoms : sequence of BaseGeometry
        源坐标系上的一组几何对象.

    crs_from : CRS
        源坐标系.

    crs_to : CRS
        目标坐标系.

    Returns
    -------
    geoms : list of BaseGeometry
        目标坐标系上的一组几何对象.
    '''
    # 坐标系相同时直接复制.
    if crs_from == crs_to:
        return [type(geom)(geom) for geom in geoms]

    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    def func(coords: CoordinateSequence) -> np.ndarray:
        coords = np.asarray(coords)
        return np.column_stack(
            transformer.transform(coords[:, 0], coords[:, 1])
        ).squeeze()
    return [_transform(func, geom) for geom in geoms]

def transform_geometry(
    geom: BaseGeometry,
    crs_from: CRS,
    crs_to: CRS
) -> BaseGeometry:
    '''
    将一个几何对象从crs_from坐标系变换到crs_to坐标系上.

    基于pyproj.Transformer实现. 相比cartopy.crs.Projection.project_geometry
    速度更快, 但当几何对象跨越坐标系边界时可能产生错误的连线.

    Parameters
    ----------
    geom : BaseGeometry
        源坐标系上的几何对象.

    crs_from : CRS
        源坐标系.

    crs_to : CRS
        目标坐标系.

    Returns
    -------
    geom : BaseGeometry
        目标坐标系上的几何对象.
    '''
    return transform_geometries([geom], crs_from, crs_to)[0]