import math
from weakref import WeakValueDictionary, WeakKeyDictionary
from pathlib import PurePath
from collections.abc import Sequence
from typing import Any, Optional, Union, Literal

import numpy as np
import shapely.geometry as sgeom
from shapely.ops import unary_union
from pyproj import Geod

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.path import Path as Path
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.transforms import Bbox, Affine2D, ScaledTranslation, offset_copy
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, Colormap, ListedColormap, BoundaryNorm
from matplotlib.quiver import Quiver, QuiverKey
from matplotlib.text import Text
from matplotlib.ticker import Formatter, AutoMinorLocator
from PIL import Image

import cartopy
if cartopy.__version__ < '0.20.0':
    raise RuntimeError('cartopy版本不低于0.20.0')
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.feature_artist import _GeomKey
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.crs import CRS, PlateCarree, Mercator, _RectangularProjection
from cartopy.feature import LAND, OCEAN

import frykit.shp as fshp
from frykit import DATA_DIRPATH

# 当polygon的引用计数为零时, 弱引用会自动清理缓存.
_key_to_polygon = WeakValueDictionary()
_key_to_transformed_polygon = WeakKeyDictionary()
_transform_func = None

def enable_fast_transform() -> None:
    '''启用快速坐标变换. 可能在地图边界产生错误的连线.'''
    global _transform_func
    _transform_func = fshp.transform_geometry
    _key_to_transformed_polygon.clear()

def disable_fast_transform() -> None:
    '''关闭快速坐标变换. 变换结果更正确, 但速度可能较慢.'''
    global _transform_func
    def _transform_func(geom, crs_from, crs_to):
        return crs_to.project_geometry(geom, crs_from)
    _key_to_transformed_polygon.clear()

enable_fast_transform()

def _cached_transform_func(
    polygon: fshp.PolygonType,
    crs_from: CRS,
    crs_to: CRS
) -> fshp.PolygonType:
    '''调用_transform_func并缓存结果.'''
    if crs_from == crs_to:
        return polygon

    key = _GeomKey(polygon)
    _key_to_polygon.setdefault(key, polygon)
    mapping = _key_to_transformed_polygon.setdefault(key, {})
    value = mapping.get(crs_to)
    if value is None:
        value = _transform_func(polygon, crs_from, crs_to)
        mapping[crs_to] = value

    return value

def add_polygons(
    ax: Axes,
    polygons: Sequence[fshp.PolygonType],
    crs: Optional[CRS] = None,
    **kwargs: Any
) -> PathCollection:
    '''
    将一组多边形添加到ax上.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    polygons : sequence of PolygonType
        多边形构成的列表.

    crs : CRS, optional
        当ax是GeoAxes时会将多边形从crs表示的坐标系变换到ax所在的坐标系上.
        默认为None, 表示PlateCarree().

    **kwargs
        创建PathCollection时的关键字参数.
        例如facecolors, edgecolors, cmap, norm和array等.

    Returns
    -------
    pc : PathCollection
        代表一组多边形的集合对象.
    '''
    kwargs.setdefault('zorder', 1.5)
    array = kwargs.get('array', None)
    if array is not None and len(array) != len(polygons):
        raise ValueError('array的长度与polygons不匹配')

    # GeoAxes会对多边形做坐标变换.
    if not isinstance(ax, Axes):
        raise ValueError('ax不是Axes')
    if isinstance(ax, GeoAxes):
        crs = PlateCarree() if crs is None else crs
        trans = ax.projection._as_mpl_transform(ax)
        func = lambda x: fshp.polygon_to_path(
            _cached_transform_func(x, crs, ax.projection)
        )
    else:
        if crs is not None:
            raise ValueError('ax不是GeoAxes时crs只能为None')
        trans = ax.transData
        func = fshp.polygon_to_path

    # PathCollection比PathPatch更快
    paths = [func(polygon) for polygon in polygons]
    pc = PathCollection(paths, transform=trans, **kwargs)
    ax.add_collection(pc)

    return pc

def add_polygon(
    ax: Axes,
    polygon: fshp.PolygonType,
    crs: Optional[CRS] = None,
    **kwargs: Any
) -> PathCollection:
    '''
    将一个多边形添加到ax上.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    polygon : PolygonType
        多边形对象.

    crs : CRS, optional
        当ax是GeoAxes时会将多边形从crs表示的坐标系变换到ax所在的坐标系上.
        默认为None, 表示PlateCarree().

    **kwargs
        创建PathCollection时的关键字参数.
        例如facecolors, edgecolors, cmap, norm和array等.

    Returns
    -------
    pc : PathCollection
        只含一个多边形的集合对象.
    '''
    return add_polygons(ax, [polygon], crs, **kwargs)

def _get_boundary(ax: GeoAxes) -> sgeom.Polygon:
    '''将GeoAxes.patch转为data坐标系下的多边形.'''
    patch = ax.patch
    # 决定patch的形状. 兼容Matplotlib3.6之前的版本.
    patch.draw(ax.figure.canvas.get_renderer())
    trans = patch.get_transform() - ax.transData
    # get_path比get_verts更可靠.
    path = patch.get_path().transformed(trans)
    boundary = sgeom.Polygon(path.vertices)

    return boundary

def clip_by_polygon(
    artist: Union[Artist, Sequence[Artist]],
    polygon: fshp.PolygonType,
    crs: Optional[CRS] = None
) -> None:
    '''
    利用多边形裁剪Artist, 只显示多边形内的内容.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.

    Parameters
    ----------
    artist : Artist or sequence of Artist
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    polygon : PolygonType
        用于裁剪的多边形对象.

    crs : CRS, optional
        当Artist在GeoAxes里时会将多边形从crs表示的坐标系变换到Artist所在的坐标系上.
        默认为None, 表示PlateCarree().
    '''
    artists = artist if isinstance(artist, Sequence) else [artist]
    ax = artists[0].axes
    for i in range(1, len(artists)):
        if artists[i].axes is not ax:
            raise ValueError('一组Artist必须属于同一个Axes')

    # Axes会自动给Artist设置clipbox, 所以不会出界.
    # GeoAxes需要在data坐标系里求polygon和patch的交集.
    if not isinstance(ax, Axes):
        raise ValueError('ax不是Axes')
    if isinstance(ax, GeoAxes):
        trans = ax.projection._as_mpl_transform(ax)
        crs = PlateCarree() if crs is None else crs
        polygon = _cached_transform_func(polygon, crs, ax.projection)
        boundary = _get_boundary(ax)
        polygon = polygon & boundary
    else:
        if crs is not None:
            raise ValueError('ax不是GeoAxes时crs只能为None')
        trans = ax.transData

    path = fshp.polygon_to_path(polygon)

    # TODO:
    # - Text.clipbox的范围可能比_clippath小.
    # - 改变显示范围, 拖拽或缩放都会影响效果.
    for artist in artists:
        if isinstance(artist, Text):
            point = sgeom.Point(artist.get_position())
            if not polygon.contains(point):
                artist.set_visible(False)
        elif hasattr(artist, 'collections'):
            for collection in artist.collections:
                collection.set_clip_path(path, trans)
        else:
            artist.set_clip_path(path, trans)

def clip_by_polygons(
    artist: Union[Artist, list[Artist]],
    polygons: Sequence[fshp.PolygonType],
    crs: Optional[CRS] = None
) -> None:
    '''
    利用一组多边形裁剪Artist, 只显示多边形内的内容.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.
    不像clip_by_polygon, 该函数无法利用缓存加快二次运行的速度.

    Parameters
    ----------
    artist : Artist or list of Artist
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    polygons : sequence of PolygonType
        用于裁剪的一组多边形.

    crs : CRS, optional
        当Artist在GeoAxes里时会将多边形从crs表示的坐标系变换到Artist所在的坐标系上.
        默认为None, 表示PlateCarree().
    '''
    polygon = unary_union(polygons)
    clip_by_polygon(artist, polygon, crs)

# 缓存常用数据.
_data_cache = {}

def _set_add_cn_kwargs(kwargs: dict) -> None:
    '''初始化add_cn_xxx函数的参数.'''
    if not any(kw in kwargs for kw in ['facecolor', 'facecolors', 'fc']):
        kwargs['facecolors'] = 'none'
    if not any(kw in kwargs for kw in ['edgecolor', 'edgecolors', 'ec']):
        kwargs['edgecolors'] = 'black'

def _get_cn_border() -> fshp.PolygonType:
    '''获取中国国界并缓存结果.'''
    country = _data_cache.get('country')
    if country is None:
        country = fshp.get_cn_shp(level='国')
        _data_cache['country'] = country

    return country

def _get_cn_province() -> dict[str, fshp.PolygonType]:
    '''获取中国省界并缓存结果.'''
    mapping = _data_cache.get('province')
    if mapping is None:
        names = fshp.get_cn_province_names()
        provinces = fshp.get_cn_shp(level='省')
        mapping = dict(zip(names, provinces))
        _data_cache['province'] = mapping

    return mapping

def _get_nine_line() -> fshp.PolygonType:
    '''获取九段线并缓存结果.'''
    nine_line = _data_cache.get('nine_line')
    if nine_line is None:
        nine_line = fshp.get_nine_line()
        _data_cache['nine_line'] = nine_line

    return nine_line

def _get_land(resolution: Literal['10m', '50m', '110m']) -> fshp.PolygonType:
    '''获取全球陆地并缓存结果.'''
    mapping = _data_cache.setdefault('land', {})
    land = mapping.get(resolution)
    if land is None:
        land = unary_union(list(LAND.with_scale(resolution).geometries()))
        mapping[resolution] = land

    return land

def _get_ocean(resolution: Literal['10m', '50m', '110m']) -> fshp.PolygonType:
    '''获取全球海洋并缓存结果.'''
    mapping = _data_cache.setdefault('ocean', {})
    ocean = mapping.get(resolution)
    if ocean is None:
        ocean = unary_union(list(OCEAN.with_scale(resolution).geometries()))
        mapping[resolution] = ocean

    return ocean

def add_cn_border(ax: Axes, **kwargs: Any) -> PathCollection:
    '''
    将中国国界添加到ax上.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    **kwargs
        创建PathCollection时的关键字参数.
        例如facecolors, edgecolors, cmap, norm和array等.

    Returns
    -------
    pc : PathCollection
        只含国界的集合对象.
    '''
    country = _get_cn_border()
    _set_add_cn_kwargs(kwargs)
    pc = add_polygon(ax, country, **kwargs)

    return pc

def add_cn_province(
    ax: Axes,
    name: Optional[Union[str, Sequence[str]]] = None,
    **kwargs: Any
) -> PathCollection:
    '''
    将中国省界添加到ax上.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    name : str or sequence of str, optional
        省名, 可以是字符串或一组字符串. 默认为None, 表示添加所有省份.

    **kwargs
        创建PathCollection时的关键字参数.
        例如facecolors, edgecolors, cmap, norm和array等.

    Returns
    -------
    pc : PathCollection
        代表省界的集合对象.
    '''
    mapping = _get_cn_province()
    if name is None:
        provinces = mapping.values()
    elif isinstance(name, str):
        provinces = [mapping[name]]
    else:
        provinces = [mapping[n] for n in name]
    _set_add_cn_kwargs(kwargs)
    pc = add_polygons(ax, provinces, **kwargs)

    return pc

def add_nine_line(ax: Axes, **kwargs: Any) -> PathCollection:
    '''
    将九段线添加到ax上.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    **kwargs
        创建PathCollection时的关键字参数.
        例如facecolors, edgecolors, cmap, norm和array等.

    Returns
    -------
    pc : PathCollection
        只含九段线的集合对象.
    '''
    nine_line = _get_nine_line()
    _set_add_cn_kwargs(kwargs)
    pc = add_polygon(ax, nine_line, **kwargs)

    return pc

def clip_by_cn_border(artist: Union[Artist, list[Artist]]) -> None:
    '''
    利用中国国界裁剪Artist.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.

    Parameters
    ----------
    artist : Artist or list of Artist
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver
    '''
    country = _get_cn_border()
    clip_by_polygon(artist, country)

def clip_by_cn_province(artist: Union[Artist, list[Artist]], name: str) -> None:
    '''
    利用中国省界裁剪Artist.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.

    Parameters
    ----------
    artist : Artist or list of Artist
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    name : str
        省名. 与add_cn_province不同, 只能指定单个省.
    '''
    mapping = _get_cn_province()
    if not isinstance(name, str):
        raise ValueError('name只能为字符串')
    province = mapping[name]
    clip_by_polygon(artist, province)

def clip_by_land(
    artist: Union[Artist, list[Artist]],
    resolution: Literal['10m', '50m', '110m'] = '110m'
) -> None:
    '''
    利用陆地边界裁剪Artist.

    需要Cartopy自动下载数据, 分辨率太高时运行较慢.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.

    Parameters
    ----------
    artist : Artist or list of Artist
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    resolution : {'110m', '50m', '10m'}, optional
        陆地数据的分辨率. 默认为'110m'.
    '''
    land = _get_land(resolution)
    clip_by_polygon(artist, land)

def clip_by_ocean(
    artist: Union[Artist, list[Artist]],
    resolution: Literal['10m', '50m', '110m'] = '110m'
) -> None:
    '''
    利用海洋边界裁剪Artist.

    需要Cartopy自动下载数据, 分辨率太高时运行较慢.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.

    Parameters
    ----------
    artist : Artist or list of Artist
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    resolution : {'110m', '50m', '10m'}, optional
        海洋数据的分辨率. 默认为'110m'.
    '''
    ocean = _get_ocean(resolution)
    clip_by_polygon(artist, ocean)

def _set_rectangular(
    ax: GeoAxes,
    extents: Optional[Any] = None,
    xticks: Optional[Any] = None,
    yticks: Optional[Any] = None,
    nx: int = 0,
    ny: int = 0,
    xformatter: Optional[Formatter] = None,
    yformatter: Optional[Formatter] = None
) -> None:
    '''设置矩形投影的GeoAxes的范围和刻度.'''
    # 默认formatter.
    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()

    # 设置x轴主次刻度.
    crs = PlateCarree()
    if xticks is not None:
        ax.set_xticks(xticks, crs=crs)
        if nx > 0:
            ax.xaxis.set_minor_locator(AutoMinorLocator(nx + 1))
        ax.xaxis.set_major_formatter(xformatter)

    # 设置y轴主次刻度.
    if yticks is not None:
        ax.set_yticks(yticks, crs=crs)
        if ny > 0:
            ax.yaxis.set_minor_locator(AutoMinorLocator(ny + 1))
        ax.yaxis.set_major_formatter(yformatter)

    # 后调用set_extent, 防止刻度拓宽显示范围.
    if extents is None:
        ax.set_global()
    else:
        ax.set_extent(extents, crs)

def _set_non_rectangular(
    ax: GeoAxes,
    extents: Optional[Any] = None,
    xticks: Optional[Any] = None,
    yticks: Optional[Any] = None,
    nx: int = 0,
    ny: int = 0,
    xformatter: Optional[Formatter] = None,
    yformatter: Optional[Formatter] = None
) -> None:
    '''设置非矩形投影的GeoAxes的范围和刻度.'''
    # 默认formatter.
    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()

    # 先设置范围, 使地图边框呈矩形.
    crs = PlateCarree()
    ax.set_extent(extents, crs)

    # 获取ax的经纬度范围.
    eps = 1
    npt = 100
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = np.linspace(x0, x1, npt)
    y = np.linspace(y0, y1, npt)
    X, Y = np.meshgrid(x, y)
    coords = crs.transform_points(ax.projection, X.ravel(), Y.ravel())
    lon, lat = coords[:, 0], coords[:, 1]
    lon0 = np.nanmin(lon) - eps
    lon1 = np.nanmax(lon) + eps
    lat0 = np.nanmin(lat) - eps
    lat1 = np.nanmax(lat) + eps

    # 以经线与上下axis的交点作为刻度.
    if xticks is not None:
        bottom_ticklocs, bottom_ticklabels = [], []
        top_ticklocs, top_ticklabels = [], []
        bottom_axis = sgeom.LineString([(x0, y0), (x1, y0)])
        top_axis = sgeom.LineString([(x0, y1), (x1, y1)])
        lat = np.linspace(lat0, lat1, npt)
        for xtick in xticks:
            if xtick < lon0 or xtick > lon1:
                continue
            lon = np.full_like(lat, xtick)
            line = sgeom.LineString(np.column_stack([lon, lat]))
            line = _transform_func(line, crs, ax.projection)
            point = bottom_axis.intersection(line)
            if isinstance(point, sgeom.Point) and not point.is_empty:
                bottom_ticklocs.append(point.x)
                bottom_ticklabels.append(xformatter(xtick))
            point = top_axis.intersection(line)
            if isinstance(point, sgeom.Point) and not point.is_empty:
                top_ticklocs.append(point.x)
                top_ticklabels.append(xformatter(xtick))

        # 让上下axis的刻度不同.
        ax.set_xticks(bottom_ticklocs + top_ticklocs)
        ax.set_xticklabels(bottom_ticklabels + top_ticklabels)
        ind = len(bottom_ticklabels)
        for tick in ax.xaxis.get_major_ticks()[:ind]:
            tick.tick2line.set_alpha(0)
            tick.label2.set_alpha(0)
        for tick in ax.xaxis.get_major_ticks()[ind:]:
            tick.tick1line.set_alpha(0)
            tick.label1.set_alpha(0)

    # 以纬线与左右axis的交点作为刻度.
    if yticks is not None:
        left_ticklocs, left_ticklabels = [], []
        right_ticklocs, right_ticklabels = [], []
        left_axis = sgeom.LineString([(x0, y0), (x0, y1)])
        right_axis = sgeom.LineString([(x1, y0), (x1, y1)])
        lon = np.linspace(lon0, lon1, npt)
        for ytick in yticks:
            if ytick < lat0 or ytick > lat1:
                continue
            lat = np.full_like(lon, ytick)
            line = sgeom.LineString(np.column_stack([lon, lat]))
            line = _transform_func(line, crs, ax.projection)
            point = left_axis.intersection(line)
            if isinstance(point, sgeom.Point) and not point.is_empty:
                left_ticklocs.append(point.y)
                left_ticklabels.append(yformatter(ytick))
            point = right_axis.intersection(line)
            if isinstance(point, sgeom.Point) and not point.is_empty:
                right_ticklocs.append(point.y)
                right_ticklabels.append(yformatter(ytick))

        # 让左右axis的刻度不同.
        ax.set_yticks(left_ticklocs + right_ticklocs)
        ax.set_yticklabels(left_ticklabels + right_ticklabels)
        ind = len(left_ticklabels)
        for tick in ax.yaxis.get_major_ticks()[:ind]:
            tick.tick2line.set_alpha(False)
            tick.label2.set_alpha(False)
        for tick in ax.yaxis.get_major_ticks()[ind:]:
            tick.tick1line.set_alpha(False)
            tick.label1.set_alpha(False)

def set_extent_and_ticks(
    ax: GeoAxes,
    extents: Optional[Any] = None,
    xticks: Optional[Any] = None,
    yticks: Optional[Any] = None,
    nx: int = 0,
    ny: int = 0,
    xformatter: Optional[Formatter] = None,
    yformatter: Optional[Formatter] = None
) -> None:
    '''
    设置GeoAxes的范围和刻度.

    支持矩形投影和显示范围为矩形的非矩形投影.
    如果在非矩形投影上的效果存在问题, 建议换用GeoAxes.gridlines.

    建议在设置刻度属性(例如tick_params)之后再调用该函数.

    Parameters
    ----------
    ax : GeoAxes
        目标GeoAxes.

    extents : (4,) array_like, optional
        经纬度范围[lon0, lon1, lat0, lat1]. 默认为None, 表示全球范围.

    xticks : array_like, optional
        经度主刻度的坐标. 默认为None, 表示不设置.

    yticks : array_like, optional
        纬度主刻度的坐标. 默认为None, 表示不设置.

    nx : int, optional
        经度主刻度之间次刻度的个数. 默认为0.
        当投影为非矩形投影或经度不是等距分布时, 请不要进行设置.

    ny : int, optional
        纬度主刻度之间次刻度的个数. 默认为0.
        当投影为非矩形投影或纬度不是等距分布时, 请不要进行设置.

    xformatter : Formatter, optional
        经度刻度标签的Formatter. 默认为None, 表示无参数的LongitudeFormatter.

    yformatter : Formatter, optional
        纬度刻度标签的Formatter. 默认为None, 表示无参数的LatitudeFormatter.
    '''
    if not isinstance(ax, GeoAxes):
        raise ValueError('ax必须是GeoAxes')
    if isinstance(ax.projection, (_RectangularProjection, Mercator)):
        func = _set_rectangular
    else:
        func = _set_non_rectangular
    func(ax, extents, xticks, yticks, nx, ny, xformatter, yformatter)

def _create_kwargs(kwargs: Optional[dict]) -> dict:
    '''创建参数字典.'''
    return {} if kwargs is None else kwargs.copy()

def add_quiver_legend(
    Q: Quiver,
    U: float,
    units: str = 'm/s',
    width: float = 0.15,
    height: float = 0.15,
    loc: Literal[
        'bottom left', 'bottom right', 'top left', 'top right'
    ] = 'bottom right',
    rect_kwargs: Optional[dict] = None,
    key_kwargs: Optional[dict] = None
) -> tuple[Rectangle, QuiverKey]:
    '''
    为Axes.quiver的结果添加图例.

    图例由背景方框patch和风箭头key组成.
    key下方有形如'{U} {units}'的标签.

    Parameters
    ----------
    Q : Quiver
        Axes.quiver返回的结果.

    U : float
        key的长度.

    units : str, optional
        key标签的单位. 默认为'm/s'.

    width : float, optional
        方框的宽度. 基于Axes坐标, 默认为0.15

    height : float, optional
        方框的高度. 基于Axes坐标, 默认为0.15

    loc : {'bottom left', 'bottom right', 'top left', 'top right'}, optional
        将图例摆放在四个角落中的哪一个. 默认为'bottom right'.

    rect_kwargs : dict, optional
        方框的参数. 例如facecolor, edgecolor, linewidth等.

    key_kwargs : dict, optional
        quiverkey的参数. 例如labelsep, fontproperties等.

    Returns
    -------
    rect : Rectangle
        图例方框对象.

    qk : QuiverKey
        风箭头和标签的对象.
    '''
    # 决定legend的位置.
    if loc == 'bottom left':
        x = width / 2
        y = height / 2
    elif loc == 'bottom right':
        x = 1 - width / 2
        y = height / 2
    elif loc == 'top left':
        x = width / 2
        y = 1 - height / 2
    elif loc == 'top right':
        x = 1 - width / 2
        y = 1 - height / 2
    else:
        raise ValueError('loc参数错误')

    # 初始化参数.
    rect_kwargs = _create_kwargs(rect_kwargs)
    if 'facecolor' not in rect_kwargs and 'fc' not in rect_kwargs:
        rect_kwargs['facecolor'] = 'white'
    if 'edgecolor' not in rect_kwargs and 'ec' not in rect_kwargs:
        rect_kwargs['edgecolor'] = 'black'
    if 'linewidth' not in rect_kwargs and 'lw' not in rect_kwargs:
        rect_kwargs['linewidth'] = 0.8
    rect_kwargs.setdefault('zorder', 3)
    key_kwargs = _create_kwargs(key_kwargs)

    # 在ax上添加patch.
    ax = Q.axes
    rect = Rectangle(
        (x - width / 2, y - height / 2), width, height,
        transform=ax.transAxes,
        **rect_kwargs
    )
    ax.add_patch(rect)

    # 先创建QuiverKey对象.
    qk = ax.quiverkey(
        Q, x, y, U,
        label=f'{U} {units}',
        labelpos='S',
        **key_kwargs
    )
    # 在参数中设置zorder无效.
    zorder = key_kwargs.get('zorder', 3)
    qk.set_zorder(zorder)

    # 再将qk调整至patch的中心.
    fontsize = qk.text.get_fontsize() / 72
    dy = (qk._labelsep_inches + fontsize) / 2
    trans = offset_copy(ax.transAxes, ax.figure, 0, dy)
    qk._set_transform = lambda: None  # 无效类方法.
    qk.set_transform(trans)

    return rect, qk

def add_compass(
    ax: Axes,
    x: float,
    y: float,
    angle: Optional[float] = None,
    size: float = 20,
    style: Literal['arrow', 'star'] = 'arrow',
    path_kwargs: Optional[dict] = None,
    text_kwargs: Optional[dict] = None
) -> tuple[PathCollection, Text]:
    '''
    向Axes添加指北针.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    x, y : float
        指北针的横纵坐标. 基于axes坐标系.

    angle : float
        指北针的方向, 从x轴逆时针方向算起, 单位为度. 默认为None.
        当ax是GeoAxes时默认自动计算角度, 否则默认表示90度.

    size : float, optional
        指北针的大小, 单位为点(point). 默认为20.

    style : {'arrow', 'star'}, optional
        指北针的造型. 默认为'arrow'.

    path_kwargs : dict, optional
        指北针的PathCollection的关键字参数.
        例如facecolors, edgecolors, linewidth等.
        默认为None, 表示使用默认参数.

    text_kwargs : dict, optional
        绘制指北针N字的关键字参数.
        例如fontsize, fontweight和fontfamily等.
        默认为None, 表示使用默认参数.

    Returns
    -------
    pc : PathCollection
        表示指北针的Collection对象.

    text : Text
        指北针N字对象.
    '''
    # 初始化箭头参数.
    path_kwargs = _create_kwargs(path_kwargs)
    if not any(kw in path_kwargs for kw in ['facecolor', 'facecolors', 'fc']):
        path_kwargs['facecolors'] = ['black', 'white']
    if not any(kw in path_kwargs for kw in ['edgecolor', 'edgecolors', 'ec']):
        path_kwargs['edgecolors'] = 'black'
    if not any(kw in path_kwargs for kw in ['linewidth', 'linewidths', 'lw']):
        path_kwargs['linewidths'] = 1
    path_kwargs.setdefault('zorder', 3)
    path_kwargs.setdefault('clip_on', False)

    # 初始化文字参数.
    text_kwargs = _create_kwargs(text_kwargs)
    if 'fontsize' not in text_kwargs and 'size' not in text_kwargs:
        text_kwargs['fontsize'] = size / 1.5

    # 计算(lon, lat)到(lon, lat + 1)的角度.
    # 当(x, y)超出Axes范围时会计算出无意义的角度.
    if angle is None:
        if isinstance(ax, GeoAxes):
            crs = PlateCarree()
            axes_to_data = ax.transAxes - ax.transData
            x0, y0 = axes_to_data.transform((x, y))
            lon0, lat0 = crs.transform_point(x0, y0, ax.projection)
            lon1, lat1 = lon0, min(lat0 + 1, 90)
            x1, y1 = ax.projection.transform_point(lon1, lat1, crs)
            angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        else:
            angle = 90

    # 指北针的大小基于物理坐标系, 旋转基于data坐标系, 平移基于axes坐标系.
    rotation = Affine2D().rotate_deg(angle)
    translation = ScaledTranslation(x, y, ax.transAxes)
    trans = ax.figure.dpi_scale_trans + rotation + translation

    # 用Path画出指北针箭头.
    head = size / 72
    if style == 'arrow':
        width = axis = head * 2 / 3
        verts1 = [(0, 0), (axis, 0), (axis - head, width / 2), (0, 0)]
        verts2 = [(0, 0), (axis - head, -width / 2), (axis, 0), (0, 0)]
        paths = [Path(verts1), Path(verts2)]
    elif style == 'star':
        width = head / 3
        axis = head + width / 2
        verts1 = [(0, 0), (axis, 0), (axis - head, width / 2), (0, 0)]
        verts2 = [(0, 0), (axis - head, -width / 2), (axis, 0), (0, 0)]
        path1 = Path(verts1)
        path2 = Path(verts2)
        paths = []
        for deg in range(0, 360, 90):
            rotation = Affine2D().rotate_deg(deg)
            paths.append(path1.transformed(rotation))
            paths.append(path2.transformed(rotation))
    else:
        raise ValueError('style参数错误')

    # 添加指北针.
    pc = PathCollection(paths, transform=trans, **path_kwargs)
    ax.add_collection(pc)

    # 添加N字.
    pad = head / 3
    text = ax.text(
        axis + pad, 0, 'N',
        ha='center', va='center',
        rotation=angle - 90,
        transform=trans,
        **text_kwargs,
    )

    return pc, text

def add_map_scale(
    ax: GeoAxes,
    x: float,
    y: float,
    length: float = 1000,
    units: Literal['m', 'km'] = 'km'
) -> Axes:
    '''
    向GeoAxes添加地图比例尺.

    用Axes模拟比例尺, 刻度可以直接通过scale.set_xticks进行修改.

    Parameters
    ----------
    ax : GeoAxes
        目标GeoAxes.

    x, y : float
        比例尺左端的横纵坐标. 基于axes坐标系.

    length : float, optional
        比例尺的长度. 默认为1000.

    units : {'m', 'km'}, optional
        比例尺长度的单位. 默认为'km'.

    Returns
    -------
    scale : Axes
        表示比例尺的Axes对象.
    '''
    if not isinstance(ax, GeoAxes):
        raise ValueError("ax只能是GeoAxes")

    if units == 'km':
        unit = 1000
    elif units == 'm':
        unit = 1
    else:
        raise ValueError('units参数错误')

    # 取地图中心一小段水平线计算单位投影坐标的长度.
    crs = PlateCarree()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    dx = (xmax - xmin) / 10
    x0 = xmid - dx / 2
    x1 = xmid + dx / 2
    lon0, lat0 = crs.transform_point(x0, ymid, ax.projection)
    lon1, lat1 = crs.transform_point(x1, ymid, ax.projection)
    geod = Geod(ellps='WGS84')
    dr = geod.inv(lon0, lat0, lon1, lat1)[2] / unit
    dxdr = dx / dr

    # axes坐标转为data坐标.
    axes_to_data = ax.transAxes - ax.transData
    x, y = axes_to_data.transform((x, y))
    width = length * dxdr

    # 避免全局的rc设置影响刻度的样式.
    bounds = [x, y, width, 1e-4 * width]
    with plt.style.context('default'):
        scale = ax.inset_axes(bounds, transform=ax.transData)
    scale.tick_params(
        left=False, labelleft=False,
        bottom=False, labelbottom=False,
        top=True, labeltop=True,
        labelsize='small'
    )
    scale.set_xlabel(units, fontsize='medium')
    scale.set_xlim(0, length)

    return scale

def add_box(
    ax: Axes,
    extents: Any,
    steps: int = 100,
    **kwargs: Any
) -> PathPatch:
    '''
    在Axes上添加一个方框.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    extents : (4,) array_like
        方框范围[x0, x1, y0, y1].

    steps: int
        在方框上重采样出N*steps个点. 默认为 100.
        当ax是GeoAxes且指定transform关键字时能保证方框的平滑.

    **kwargs
        创建PathPatch对象的关键字参数.
        例如linewidth, edgecolor, facecolor和transform等.

    Returns
    -------
    patch : PathPatch
        方框对象.
    '''
    # 初始化参数.
    if 'facecolor' not in kwargs or 'fc' not in kwargs:
        kwargs['facecolor'] = 'none'
    if 'edgecolor' not in kwargs or 'ec' not in kwargs:
        kwargs['edgecolor'] = 'r'

    # 添加Patch.
    x0, x1, y0, y1 = extents
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    path = Path(verts).interpolated(steps)
    patch = PathPatch(path, **kwargs)
    ax.add_patch(patch)

    return patch

def load_test_nc():
    '''读取测试用的nc文件. 需要安装xarray和NetCDF4.'''
    import xarray as xr
    filepath = DATA_DIRPATH / 'test.nc'
    ds = xr.load_dataset(str(filepath))

    return ds

# TODO: inset_axes实现.
def move_axes_to_corner(
    ax: Axes,
    ref_ax: Axes,
    shrink: float = 0.4,
    loc: Literal[
        'bottom left', 'bottom right', 'top left', 'top right'
    ] = 'bottom right'
) -> None:
    '''
    讲ax等比例缩小并放置在ref_ax的角落位置.

    Parameters
    ----------
    ax : Axes
        目标Axes. 若为GeoAxes, 需要提前调用ax.set_extent确定大小.

    ref_ax : Axes
        作为参考的Axes.

    shrink : float, optional
        缩小倍数. 默认为0.4.

    loc : {'bottom left', 'bottom right', 'top left', 'top right'}, optional
        指定放置在哪个角落. 默认为'bottom right'.
    '''
    bbox = ax.get_position()
    ref_bbox = ref_ax.get_position()
    # 使shrink=1时ax与ref_ax等宽或等高.
    if bbox.width > bbox.height:
        ratio = ref_bbox.width / bbox.width * shrink
    else:
        ratio = ref_bbox.height / bbox.height * shrink
    width = bbox.width * ratio
    height = bbox.height * ratio

    # 可选四个角落位置.
    if loc == 'bottom left':
        x0 = ref_bbox.x0
        x1 = ref_bbox.x0 + width
        y0 = ref_bbox.y0
        y1 = ref_bbox.y0 + height
    elif loc == 'bottom right':
        x0 = ref_bbox.x1 - width
        x1 = ref_bbox.x1
        y0 = ref_bbox.y0
        y1 = ref_bbox.y0 + height
    elif loc == 'top left':
        x0 = ref_bbox.x0
        x1 = ref_bbox.x0 + width
        y0 = ref_bbox.y1 - height
        y1 = ref_bbox.y1
    elif loc == 'top right':
        x0 = ref_bbox.x1 - width
        x1 = ref_bbox.x1
        y0 = ref_bbox.y1 - height
        y1 = ref_bbox.y1
    else:
        raise ValueError('loc参数错误')
    new_bbox = Bbox.from_extents(x0, y0, x1, y1)
    ax.set_position(new_bbox)

def add_side_axes(
    ax: Any,
    loc: Literal['left', 'right', 'bottom', 'top'],
    pad : float,
    depth: float
) -> Axes:
    '''
    在原有的Axes旁边新添一个等高或等宽的Axes并返回该对象.

    Parameters
    ----------
    ax : Axes or array_like of Axes
        原有的Axes, 也可以是一组Axes构成的数组.

    loc : {'left', 'right', 'bottom', 'top'}
        新Axes相对于旧Axes的位置.

    pad : float
        新旧Axes的间距. 基于Figure坐标系.

    depth : float
        新Axes的宽度或高度. 基于Figure坐标系.

    Returns
    -------
    side_ax : Axes
        新Axes对象.
    '''
    # 获取一组Axes的位置.
    axs = np.atleast_1d(ax).ravel()
    bbox = Bbox.union([ax.get_position() for ax in axs])

    # 可选四个方向.
    if loc == 'left':
        x0 = bbox.x0 - pad - depth
        x1 = x0 + depth
        y0 = bbox.y0
        y1 = bbox.y1
    elif loc == 'right':
        x0 = bbox.x1 + pad
        x1 = x0 + depth
        y0 = bbox.y0
        y1 = bbox.y1
    elif loc == 'bottom':
        x0 = bbox.x0
        x1 = bbox.x1
        y0 = bbox.y0 - pad - depth
        y1 = y0 + depth
    elif loc == 'top':
        x0 = bbox.x0
        x1 = bbox.x1
        y0 = bbox.y1 + pad
        y1 = y0 + depth
    else:
        raise ValueError('loc参数错误')
    side_bbox = Bbox.from_extents(x0, y0, x1, y1)
    side_ax = axs[0].figure.add_axes(side_bbox)

    return side_ax

def get_cross_section_xticks(
    lon: Any,
    lat: Any,
    ntick: int = 6,
    lon_formatter: Optional[Formatter] = None,
    lat_formatter: Optional[Formatter] = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    '''
    返回垂直截面图所需的横坐标, 刻度位置和刻度标签.

    用经纬度的欧式距离表示横坐标, 在横坐标上取ntick个等距的刻度,
    利用线性插值计算每个刻度对应的经纬度值并用作刻度标签.

    Parameters
    ----------
    lon : (npt,) array_like
        横截面对应的经度数组.

    lat : (npt,) array_like
        横截面对应的纬度数组.

    ntick : int, optional
        刻度的数量. 默认为6.

    lon_formatter : Formatter, optional
        刻度标签里经度的Formatter, 用来控制字符串的格式.
        默认为None, 表示LongitudeFormatter.

    lat_formatter : Formatter, optional
        刻度标签里纬度的Formatter. 用来控制字符串的格式.
        默认为None, 表示LatitudeFormatter.

    Returns
    -------
    x : (npt,) ndarray
        横截面的横坐标.

    xticks : (ntick,) ndarray
        刻度的横坐标.

    xticklabels : (ntick,) list of str
        用经纬度表示的刻度标签.
    '''
    # 线性插值计算刻度的经纬度值.
    npt = len(lon)
    if npt <= 1:
        raise ValueError('lon和lat至少有2个元素')
    dlon = lon - lon[0]
    dlat = lat - lat[0]
    x = np.hypot(dlon, dlat)
    xticks = np.linspace(x[0], x[-1], ntick)
    tlon = np.interp(xticks, x, lon)
    tlat = np.interp(xticks, x, lat)

    # 获取字符串形式的刻度标签.
    xticklabels = []
    if lon_formatter is None:
        lon_formatter = LongitudeFormatter(number_format='.1f')
    if lat_formatter is None:
        lat_formatter = LatitudeFormatter(number_format='.1f')
    for i in range(ntick):
        lon_label = lon_formatter(tlon[i])
        lat_label = lat_formatter(tlat[i])
        xticklabels.append(lon_label + '\n' + lat_label)

    return x, xticks, xticklabels

def make_qualitative_cmap(colors: Any) -> tuple[
    ListedColormap, Normalize, np.ndarray
]:
    '''
    创建一组定性的colormap和norm, 同时返回刻度位置.

    Parameters
    ----------
    colors : (N,) list or (N, 3) or (N, 4) array_like
        colormap所含的颜色. 可以为含有颜色的列表或RGB(A)数组.

    Returns
    -------
    cmap : ListedColormap
        创建的colormap.

    norm : Normalize
        创建的norm. N个颜色对应于0~N-1范围的数据.

    ticks : (N,) ndarray
        colorbar刻度的坐标.
    '''
    N = len(colors)
    cmap = ListedColormap(colors)
    norm = Normalize(vmin=-0.5, vmax=N-0.5)
    ticks = np.arange(N)

    return cmap, norm, ticks

def get_aod_cmap() -> ListedColormap:
    '''返回适用于AOD的cmap.'''
    filepath = DATA_DIRPATH / 'NEO_modis_aer_od.csv'
    rgb = np.loadtxt(str(filepath), delimiter=',') / 256
    cmap = ListedColormap(rgb)

    return cmap

class CenteredBoundaryNorm(BoundaryNorm):
    '''将vcenter所在的bin映射到cmap中间的BoundaryNorm.'''
    def __init__(
        self,
        boundaries: Any,
        vcenter: float = 0,
        clip: bool = False
    ) -> None:
        super().__init__(boundaries, len(boundaries) - 1, clip)
        boundaries = np.asarray(boundaries)
        self.N1 = np.count_nonzero(boundaries < vcenter)
        self.N2 = np.count_nonzero(boundaries > vcenter)
        if self.N1 < 1 or self.N2 < 1:
            raise ValueError('vcenter两侧至少各有一条边界')

    def __call__(
        self,
        value: Any,
        clip: Optional[bool] = None
    ) -> np.ma.MaskedArray:
        # 将BoundaryNorm的[0, N-1]又映射到[0.0, 1.0]内.
        result = super().__call__(value, clip)
        if self.N1 + self.N2 == self.N - 1:
            result = np.ma.where(
                result < self.N1,
                result / (2 * self.N1),
                (result - self.N1 + self.N2 + 1) / (2 * self.N2)
            )
        else:
            # 当result是MaskedArray时除以零不会报错.
            result = np.ma.where(
                result < self.N1,
                result / (2 * (self.N1 - 1)),
                (result - self.N1 + self.N2) / (2 * (self.N2 - 1))
            )

        return result

def plot_colormap(
    cmap: Colormap,
    norm: Optional[Normalize] = None,
    ax: Optional[Axes] = None
) -> Colorbar:
    '''快速展示一条colormap.'''
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 1))
        fig.subplots_adjust(bottom=0.5)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(mappable, cax=ax, orientation='horizontal')

    return cbar

def letter_axes(axes: Any, x: float, y: float, **kwargs: Any) -> None:
    '''
    给一组Axes按顺序标注字母.

    Parameters
    ----------
    axes : array_like of Axes
        目标Axes的数组.

    x : float or array_like
        字母的横坐标, 基于Axes单位.
        可以为标量或数组, 数组形状需与axes相同.

    y : float or array_like
        字母的纵坐标. 基于Axes单位.
        可以为标量或数组, 数组形状需与axes相同.

    y : float or array_like
        可以为标量或数组, 数组形状需与axes相同.

    **kwargs
        调用text时的关键字参数.
        例如fontsize, fontfamily和color等.
    '''
    axes = np.atleast_1d(axes)
    x = np.full_like(axes, x) if np.isscalar(x) else np.asarray(x)
    y = np.full_like(axes, y) if np.isscalar(y) else np.asarray(y)
    for i, (ax, xi, yi) in enumerate(zip(axes.flat, x.flat, y.flat)):
        letter = chr(97 + i)
        ax.text(
            xi, yi, f'({letter})',
            ha='center',
            va='center',
            transform=ax.transAxes,
            **kwargs
        )

def make_gif(
    img_filepaths: Union[Sequence[str], Sequence[PurePath]],
    gif_filepath: Union[str, PurePath],
    duration: int = 500,
    loop: int = 0,
    optimize: bool = False
) -> None:
    '''
    制作GIF图.

    Parameters
    ----------
    img_filepaths : sequence of str or sequence of PurePath
        图片路径的列表. 要求至少含两个元素.

    gif_filepath : str or PurePath
        输出GIF图片的路径.

    duration : int or list or tuple, optional
        每一帧的持续时间, 以毫秒为单位. 也可以用列表或元组分别指定每一帧的持续时间.
        默认为500ms=0.5s.

    loop : int, optional
        GIF图片循环播放的次数. 默认无限循环.

    optimize : bool, optional
        尝试压缩GIF图片的调色板.
    '''
    if len(img_filepaths) < 2:
        raise ValueError('至少需要两张图片')

    images = [Image.open(str(filepath)) for filepath in img_filepaths]
    images[0].save(
        str(gif_filepath),
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=optimize
    )