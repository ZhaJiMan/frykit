import math
import json
from pathlib import Path
from itertools import chain

import numpy as np
import shapefile
import shapely.geometry as sgeom
from shapely.prepared import prep
from pyproj import Transformer
import matplotlib.path as mpath

def _to_geom(shapeRec):
    '''将shapeRecord转为几何对象.'''
    return sgeom.shape(shapeRec.shape)

def _to_dict(shapeRec):
    '''将shapeRecord转为字典.'''
    d = shapeRec.record.as_dict()
    d['geometry'] = _to_geom(shapeRec)
    return d

dirpath_shp = Path(__file__).parent / 'data' / 'shp'
def get_cnshp(level='国', province=None, city=None, as_dict=False):
    '''
    获取中国行政区划的多边形. 支持国界, 省界和市界.

    接口仿照cnmaps.maps.get_adm_maps
    数据源: https://github.com/GaryBikini/ChinaAdminDivisonSHP
    使用了PRCoords库从GCJ-02坐标变换到了WGS-84坐标.

    用法:
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

    province : str, optional
        省名. 默认不指定.

    city : str, optional
        市名. 默认不指定.

    as_dict : bool, optional
        是否以字典形式返回结果. 默认直接返回多边形对象.
        字典的键包括:
        - cn_name, pr_name, ct_name: 国省市名
        - cn_adcode, pr_adcode, ct_adcode: 国省市的区划代码
        - geometry: 多边形对象

    Returns
    -------
    shps : Polygon or list of Polygon, dict or list of dict
        行政区划的多边形或多边形构成的列表.
    '''
    if level == '国':
        filename = 'country.shp'
    elif level == '省':
        filename = 'province.shp'
    elif level == '市':
        filename = 'city.shp'
    else:
        raise ValueError('level参数错误')
    filepath_shp = dirpath_shp / filename
    to = _to_dict if as_dict else _to_geom

    # 利用提前制作的index.json提高查找速度.
    filepath_index = dirpath_shp / 'index.json'
    with open(str(filepath_index), 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    if level == '国':
        with shapefile.Reader(str(filepath_shp)) as reader:
            return to(reader.shapeRecord(0))

    if level == '省':
        pr_name_to_pr_index = mapping['pr_name_to_pr_index']
        if city is not None:
            raise ValueError("level='省'时不能指定市")
        if province is None:
            index = None
        elif isinstance(province, (list, tuple)):
            index = [pr_name_to_pr_index[name] for name in province]
        else:
            index = pr_name_to_pr_index[province]

    if level == '市':
        ct_name_to_ct_index = mapping['ct_name_to_ct_index']
        pr_name_to_ct_indices = mapping['pr_name_to_ct_indices']
        if province is not None and city is not None:
            raise ValueError("level='市'时不能同时指定省和市")
        if province is None and city is None:
            index = None
        if province is not None:
            if isinstance(province, (list, tuple)):
                lists = [pr_name_to_ct_indices[name] for name in province]
                index = list(chain.from_iterable(lists))
            else:
                index = pr_name_to_ct_indices[province]
        if city is not None:
            if isinstance(city, (list, tuple)):
                index = [ct_name_to_ct_index[name] for name in city]
            else:
                index = ct_name_to_ct_index[city]

    with shapefile.Reader(str(filepath_shp)) as reader:
        if index is None:
            return [to(shapeRec) for shapeRec in reader.shapeRecords()]
        elif isinstance(index, list):
            return [to(reader.shapeRecord(i)) for i in index]
        else:
            return to(reader.shapeRecord(index))

def get_nine_line(as_dict=False):
    '''
    获取九段线的多边形.

    数据源: http://datav.aliyun.com/portal/school/atlas/area_selector
    使用了PRCoords库从GCJ-02坐标变换到了WGS-84坐标.

    Parameters
    ----------
    as_dict : bool, optional
        是否以字典形式返回结果. 默认直接返回多边形对象.

    Returns
    -------
    nine_line : MultiPolygon
        用十个多边形表示的九段线.
    '''
    to = _to_dict if as_dict else _to_geom
    filepath_shp = dirpath_shp / 'nine_line.shp'
    with shapefile.Reader(str(filepath_shp)) as reader:
        return to(reader.shapeRecord(0))

def simplify_province_name(name):
    '''简化省名到2或3个字.'''
    if name.startswith('内蒙古') or name.startswith('黑龙江'):
        return name[:3]
    else:
        return name[:2]

def _ring_codes(n):
    '''为长度为n的环生成codes.'''
    codes = [mpath.Path.LINETO] * n
    codes[0] = mpath.Path.MOVETO
    codes[-1] = mpath.Path.CLOSEPOLY

    return codes

def polygon_to_path(polygon):
    '''将Polygon或MultiPolygon转为Path.'''
    if isinstance(polygon, sgeom.Polygon):
        polygons = [polygon]
    elif isinstance(polygon, sgeom.MultiPolygon):
        polygons = polygon.geoms
    else:
        raise TypeError('polygon不是多边形对象')

    # 空多边形需要占位.
    if polygon.is_empty:
        return mpath.Path([(0, 0)])

    # 用多边形含有的所有环的顶点构造Path.
    vertices, codes = [], []
    for polygon in polygons:
        for ring in [polygon.exterior] + polygon.interiors[:]:
            vertices += ring.coords[:]
            codes += _ring_codes(len(ring.coords))
    path = mpath.Path(vertices, codes)

    return path

def polygon_to_mask(polygon, x, y):
    '''
    用多边形制作掩膜(mask)数组.

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        pass

    x : array_like
        掩膜数组的横坐标. 要求形状与y相同.

    y : array_like
        掩膜数组的纵坐标. 要求形状与x相同.

    Returns
    -------
    mask : ndarray
        布尔类型的掩膜数组, 真值表示落入多边形内部. 形状与x和y相同.
    '''
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if x.shape != y.shape:
        raise ValueError('x和y的形状不匹配')
    if not isinstance(polygon, (sgeom.Polygon, sgeom.MultiPolygon)):
        raise TypeError('polygon不是多边形对象')
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

def _transform(func, geom):
    '''shapely.ops.transform的修改版, 会将坐标含无效值的对象设为空对象.'''
    if geom.is_empty:
        return type(geom)()
    if geom.type in ('Point', 'LineString', 'LinearRing'):
        coords = func(geom.coords)
        if np.isfinite(coords).all():
            return type(geom)(coords)
        else:
            return type(geom)()
    elif geom.type == 'Polygon':
        shell = func(geom.exterior.coords)
        if not np.isfinite(shell).all():
            return sgeom.Polygon()
        holes = []
        for ring in geom.interiors:
            hole = func(ring.coords)
            if np.isfinite(hole).all():
                holes.append(hole)
        return sgeom.Polygon(shell, holes)
    elif geom.type.startswith('Multi') or geom.type == 'GeometryCollection':
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

def transform_geometries(geoms, crs_from, crs_to):
    '''
    对一组几何对象的坐标进行变换, 返回变换后的对象组成的列表.

    使用pyproj.Transformer进行变换, 当几何对象跨边界时会产生错误的连线.
    此时建议使用cartopy.crs.Projection.project_geometry.

    Parameters
    ----------
    geoms : list of BaseGeometry or BaseMultipartGeometry
        源坐标系上的一组几何对象.

    crs_from : CRS
        源坐标系.

    crs_to : CRS
        目标坐标系.

    Returns
    -------
    geoms : list of BaseGeometry or BaseMultipartGeometry
        目标坐标系上的一组几何对象.
    '''
    # 坐标系相同时直接复制.
    if crs_from == crs_to:
        return [type(geom)(geom) for geom in geoms]

    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    def func(coords):
        coords = np.asarray(coords)
        return np.column_stack(
            transformer.transform(coords[:, 0], coords[:, 1])
        ).squeeze()
    return [_transform(func, geom) for geom in geoms]

def transform_geometry(geom, crs_from, crs_to):
    '''对几何对象的坐标进行变换, 返回变换后的对象.'''
    return transform_geometries([geom], crs_from, crs_to)[0]