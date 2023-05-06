from pathlib import Path
from weakref import WeakValueDictionary, WeakKeyDictionary

import numpy as np
import shapely.geometry as sgeom
from shapely.ops import unary_union
from pyproj import Geod
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.ticker import AutoMinorLocator
from matplotlib.path import Path as mPath
from matplotlib.collections import PathCollection
from matplotlib.cbook import silent_list
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.feature_artist import _GeomKey
from cartopy.feature import LAND, OCEAN
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from PIL import Image

import frykit.shp as fshp

# 当polygon的引用计数为零时, 弱引用会自动清理缓存.
_key_to_polygon_cache = WeakValueDictionary()
_key_to_transformed_cache = WeakKeyDictionary()

_transform_geometry = None

def enable_fast_transform():
    '''启用快速坐标变换. 可能在地图边界产生错误的连线.'''
    global _transform_geometry
    _transform_geometry = fshp.transform_geometry
    _key_to_transformed_cache.clear()

def disable_fast_transform():
    '''关闭快速坐标变换. 变换结果更正确, 但速度可能较慢.'''
    global _transform_geometry
    def _transform_geometry(geom, crs_from, crs_to):
        return crs_to.project_geometry(geom, crs_from)
    _key_to_transformed_cache.clear()

enable_fast_transform()

def _cached_transform(polygon, crs_from, crs_to):
    '''调用_transform_geometry并缓存结果.'''
    if crs_from == crs_to:
        return polygon

    key = _GeomKey(polygon)
    _key_to_polygon_cache.setdefault(key, polygon)
    mapping = _key_to_transformed_cache.setdefault(key, {})
    value = mapping.get(crs_to)
    if value is None:
        value = _transform_geometry(polygon, crs_from, crs_to)
        mapping[crs_to] = value

    return value

def add_polygons(ax, polygons, crs=None, **kwargs):
    '''
    将一组多边形添加到ax上.

    Parameters
    ----------
    ax : Axes or GeoAxes
        目标Axes.

    polygons : list of Polygon or list of MultiPolygon
        多边形构成的列表.

    crs : CRS, optional
        多边形所处的坐标系. 默认为ccrs.PlateCarree().
        当ax是Axes时该参数无效, 不会对多边形进行坐标变换.

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

    if crs is None:
        crs = ccrs.PlateCarree()

    # GeoAxes会对多边形做坐标变换.
    trans = ax.transData
    to = fshp.polygon_to_path
    if isinstance(ax, GeoAxes):
        trans = ax.projection._as_mpl_transform(ax)
        to = lambda x: fshp.polygon_to_path(
            _cached_transform(x, crs, ax.projection)
        )

    # PathCollection比PathPatch更快
    paths = [to(polygon) for polygon in polygons]
    pc = PathCollection(paths, transform=trans, **kwargs)
    ax.add_collection(pc)

    return pc

def add_polygon(ax, polygon, crs=None, **kwargs):
    '''
    将一个多边形添加到ax上.

    Parameters
    ----------
    ax : Axes or GeoAxes
        目标Axes.

    polygons : list of Polygon or list of MultiPolygon
        多边形构成的列表.

    crs : CRS, optional
        多边形所处的坐标系. 默认为ccrs.PlateCarree().
        当ax是Axes时该参数无效, 不会对多边形进行坐标变换.

    **kwargs
        创建PathCollection时的关键字参数.
        例如facecolors, edgecolors, cmap, norm和array等.

    Returns
    -------
    pc : PathCollection
        只含一个多边形的集合对象.
    '''
    return add_polygons(ax, [polygon], crs, **kwargs)

def _get_boundary(ax):
    '''将GeoAxes.patch转为data坐标系下的多边形.'''
    patch = ax.patch
    ax.draw_artist(patch)  # 决定patch的形状.
    trans = patch.get_transform() - ax.transData
    # get_path比get_verts更可靠.
    path = patch.get_path().transformed(trans)
    boundary = sgeom.Polygon(path.vertices)

    return boundary

def clip_by_polygon(artist, polygon, crs=None):
    '''
    利用多边形裁剪Artist, 只显示多边形内的内容.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.

    Parameters
    ----------
    artist : Artist
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    polygon : Polygon or MultiPolygon
        用于裁剪的多边形.

    crs : CRS, optional
        多边形所处的坐标系. 默认为ccrs.PlateCarree().
        当ax是Axes时该参数无效, 不会对多边形进行坐标变换.
    '''
    is_list = isinstance(artist, silent_list)
    ax = artist[0].axes if is_list else artist.axes

    if crs is None:
        crs = ccrs.PlateCarree()

    # Axes会自动给Artist设置clipbox, 所以不会出界.
    # GeoAxes需要在data坐标系里求polygon和patch的交集.
    trans = ax.transData
    if isinstance(ax, GeoAxes):
        trans = ax.projection._as_mpl_transform(ax)
        polygon = _cached_transform(polygon, crs, ax.projection)
        boundary = _get_boundary(ax)
        polygon = polygon & boundary
    path = fshp.polygon_to_path(polygon)

    # TODO:
    # - Text.clipbox的范围可能比_clippath小.
    # - 改变显示范围, 拖拽或缩放都会影响效果.
    if is_list:
        for text in artist:
            point = sgeom.Point(text.get_position())
            if not polygon.contains(point):
                text.set_visible(False)
    elif hasattr(artist, 'collections'):
        for collection in artist.collections:
            collection.set_clip_path(path, trans)
    else:
        artist.set_clip_path(path, trans)

def clip_by_polygons(artist, polygons, crs=None):
    '''
    利用一组多边形裁剪Artist, 只显示多边形内的内容.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.
    不像clip_by_polygon, 该函数无法利用缓存加快二次运行的速度.

    Parameters
    ----------
    artist : Artist
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    polygons : list of Polygon or list of MultiPolygon
        用于裁剪的一组多边形.

    crs : CRS, optional
        多边形所处的坐标系. 默认为ccrs.PlateCarree().
        当ax是Axes时该参数无效, 不会对多边形进行坐标变换.
    '''
    polygon = unary_union(polygons)
    clip_by_polygon(artist, polygon, crs)

# 缓存常用数据.
_data_cache = {}

def _set_add_cn_kwargs(kwargs):
    '''初始化add_cn_xxx函数的参数.'''
    if not any(kw in kwargs for kw in ['facecolor', 'facecolors', 'fc']):
        kwargs['facecolors'] = 'none'
    if not any(kw in kwargs for kw in ['edgecolor', 'edgecolors', 'ec']):
        kwargs['edgecolors'] = 'black'

def _get_cn_border():
    '''获取中国国界并缓存结果.'''
    country = _data_cache.get('country')
    if country is None:
        country = fshp.get_cn_shp(level='国')
        _data_cache['country'] = country

    return country

def _get_cn_province():
    '''获取中国省界并缓存结果.'''
    mapping = _data_cache.get('province')
    if mapping is None:
        names = fshp.get_cn_province_names()
        provinces = fshp.get_cn_shp(level='省')
        mapping = dict(zip(names, provinces))
        _data_cache['province'] = mapping

    return mapping

def _get_nine_line():
    '''获取九段线并缓存结果.'''
    nine_line = _data_cache.get('nine_line')
    if nine_line is None:
        nine_line = fshp.get_nine_line()
        _data_cache['nine_line'] = nine_line

    return nine_line

def _get_land(resolution):
    '''获取全球陆地并缓存结果.'''
    mapping = _data_cache.setdefault('land', {})
    land = mapping.get(resolution)
    if land is None:
        land = unary_union(list(LAND.with_scale(resolution).geometries()))
        mapping[resolution] = land

    return land

def _get_ocean(resolution):
    '''获取全球海洋并缓存结果.'''
    mapping = _data_cache.setdefault('ocean', {})
    ocean = mapping.get(resolution)
    if ocean is None:
        ocean = unary_union(list(OCEAN.with_scale(resolution).geometries()))
        mapping[resolution] = ocean

    return ocean

def add_cn_border(ax, **kwargs):
    '''
    将中国国界添加到ax上.

    Parameters
    ----------
    ax : Axes or GeoAxes
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

def add_cn_province(ax, name=None, **kwargs):
    '''
    将中国省界添加到ax上.

    Parameters
    ----------
    ax : Axes or GeoAxes
        目标Axes.

    name : str or list of str, optional
        省名, 可以是字符串或字符串构成的列表.
        默认为None, 表示添加所有省份.

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

def add_nine_line(ax, **kwargs):
    '''
    将九段线添加到ax上.

    Parameters
    ----------
    ax : Axes or GeoAxes
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

def clip_by_cn_border(artist):
    '''
    利用中国国界裁剪Artist.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.

    Parameters
    ----------
    artist : Artist
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver
    '''
    country = _get_cn_border()
    clip_by_polygon(artist, country)

def clip_by_cn_province(artist, name):
    '''
    利用中国省界裁剪Artist.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.

    Parameters
    ----------
    artist : Artist
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    name : str
        省名. 与add_cn_province不同, 只能为单个字符串.
    '''
    mapping = _get_cn_province()
    if not isinstance(name, str):
        raise ValueError('name只能为字符串')
    province = mapping[name]
    clip_by_polygon(artist, province)

def clip_by_land(artist, resolution='110m'):
    '''
    利用陆地边界裁剪Artist.

    需要Cartopy自动下载数据, 分辨率太高时运行较慢.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.

    Parameters
    ----------
    artist : Artist
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

def clip_by_ocean(artist, resolution='110m'):
    '''
    利用海洋边界裁剪Artist.

    需要Cartopy自动下载数据, 分辨率太高时运行较慢.

    裁剪后再修改GeoAxes的边界或显示范围会产生异常的效果.

    Parameters
    ----------
    artist : Artist
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
    ax, extents=None,
    xticks=None, yticks=None, nx=0, ny=0,
    xformatter=None, yformatter=None
):
    '''设置矩形投影的GeoAxes的范围和刻度.'''
    # 默认formatter.
    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()

    # 设置x轴主次刻度.
    crs = ccrs.PlateCarree()
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
    ax, extents=None,
    xticks=None, yticks=None,
    xformatter=None, yformatter=None
):
    '''设置非矩形投影的GeoAxes的范围和刻度.'''
    # 默认formatter.
    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()

    # 先设置范围, 使地图边框呈矩形.
    crs = ccrs.PlateCarree()
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
            line = _transform_geometry(line, crs, ax.projection)
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
            tick.tick2line.set_visible(False)
            tick.label2.set_visible(False)
        for tick in ax.xaxis.get_major_ticks()[ind:]:
            tick.tick1line.set_visible(False)
            tick.label1.set_visible(False)

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
            line = _transform_geometry(line, crs, ax.projection)
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
            tick.tick2line.set_visible(False)
            tick.label2.set_visible(False)
        for tick in ax.yaxis.get_major_ticks()[ind:]:
            tick.tick1line.set_visible(False)
            tick.label1.set_visible(False)

def set_extent_and_ticks(
    ax, extents=None,
    xticks=None, yticks=None, nx=0, ny=0,
    xformatter=None, yformatter=None
):
    '''
    设置GeoAxes的范围和刻度.

    支持矩形投影和显示范围为矩形的非矩形投影.
    如果在非矩形投影上的效果存在问题, 建议换用GeoAxes.gridlines.

    建议在设置刻度属性(例如tick_params)之后再调用该函数.

    Parameters
    ----------
    ax : GeoAxes
        目标GeoAxes.

    extents : 4-tuple of float, optional
        经纬度范围[lonmin, lonmax, latmin, latmax].
        默认为None, 表示全球范围.

    xticks : array_like, optional
        经度主刻度的坐标. 默认为None, 表示不设置.

    yticks : array_like, optional
        纬度主刻度的坐标. 默认为None, 表示不设置.

    nx : int, optional
        经度主刻度之间次刻度的个数. 默认为None, 表示没有次刻度.
        当投影为非矩形投影或经度不是等距分布时, 请不要进行设置.

    ny : int, optional
        纬度主刻度之间次刻度的个数. 默认为None, 表示没有次刻度.
        当投影为非矩形投影或纬度不是等距分布时, 请不要进行设置.

    xformatter : Formatter, optional
        经度刻度标签的Formatter. 默认为None, 表示无参数的LongitudeFormatter.

    yformatter : Formatter, optional
        纬度刻度标签的Formatter. 默认为None, 表示无参数的LatitudeFormatter.
    '''
    if not isinstance(ax, GeoAxes):
        raise ValueError('ax应该是GeoAxes')
    kwargs = {
        'ax': ax, 'extents': extents,
        'xticks': xticks, 'yticks': yticks, 'nx': nx, 'ny': ny,
        'xformatter': xformatter, 'yformatter': yformatter
    }

    if isinstance(ax.projection, (ccrs._RectangularProjection, ccrs.Mercator)):
        _set_rectangular(**kwargs)
    else:
        del kwargs['nx'], kwargs['ny']
        _set_non_rectangular(**kwargs)

def _create_kwargs(kwargs):
    '''创建参数字典.'''
    return {} if kwargs is None else kwargs.copy()

def add_quiver_legend(
    Q, U, units='m/s',
    width=0.15, height=0.15, loc='bottom right',
    rect_kwargs=None, key_kwargs=None
):
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
    rect = mpatches.Rectangle(
        (x - width / 2, y - height / 2), width, height,
        transform=ax.transAxes, **rect_kwargs
    )
    ax.add_patch(rect)

    # 先创建QuiverKey对象.
    qk = ax.quiverkey(
        Q, X=x, Y=y, U=U, label=f'{U} {units}',
        labelpos='S', **key_kwargs
    )
    # 在参数中设置zorder无效.
    zorder = key_kwargs.get('zorder', 3)
    qk.set_zorder(zorder)

    # 再将qk调整至patch的中心.
    fontsize = qk.text.get_fontsize() / 72
    dy = (qk._labelsep_inches + fontsize) / 2
    transform = mtransforms.offset_copy(ax.transAxes, ax.figure, 0, dy)
    qk._set_transform = lambda: None  # 无效类方法.
    qk.set_transform(transform)

    return rect, qk

def add_compass(
    ax, x, y, size=20, style='arrow',
    path_kwargs=None, text_kwargs=None
):
    '''
    向Axes添加指北针.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    x, y : float
        指北针的横纵坐标. 基于axes坐标系.

    size : float, optional
        指北针的大小, 单位为点. 默认为20.

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

    head = size / 72
    offset = mtransforms.ScaledTranslation(x, y, ax.transAxes)
    trans = ax.figure.dpi_scale_trans + offset

    # 指北针的大小基于物理坐标系, 位置基于axes坐标系.
    if style == 'arrow':
        width = axis = head * 2 / 3
        left_verts = [(0, 0), (-width / 2, axis - head), (0, axis), (0, 0)]
        right_verts = [(0, 0), (0, axis), (width / 2, axis - head), (0, 0)]
        paths = [mPath(left_verts), mPath(right_verts)]
    elif style == 'star':
        width = head / 3
        axis = head + width / 2
        left_verts = [(0, 0), (0, axis), (-width / 2, axis - head), (0, 0)]
        right_verts = [(0, 0), (width / 2, axis - head), (0, axis), (0, 0)]
        left_path = mPath(left_verts)
        right_path = mPath(right_verts)
        paths = []
        for deg in range(0, 360, 90):
            rotate = mtransforms.Affine2D().rotate_deg(deg)
            paths.append(left_path.transformed(rotate))
            paths.append(right_path.transformed(rotate))
    else:
        raise ValueError('style参数错误')

    # 添加指北针.
    pc = PathCollection(paths, transform=trans, **path_kwargs)
    ax.add_collection(pc)

    # 添加N字.
    pad = head / 10
    text = ax.text(
        0, axis + pad, 'N', ha='center', va='bottom',
        transform=trans, **text_kwargs
    )

    return pc, text

def add_map_scale(ax, x, y, length=1000, units='km'):
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

    units : {'km', 'm'}, optional
        比例尺长度的单位. 默认为'km'.

    Returns
    -------
    scale : Axes
        表示比例尺的Axes对象.
    '''
    if units == 'km':
        unit = 1000
    elif units == 'm':
        unit = 1
    else:
        raise ValueError('units参数错误')

    # 取地图中心的水平线计算单位投影坐标的长度.
    crs = ccrs.PlateCarree()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    dx = xmax - xmin
    x0 = xmid - dx / 2
    x1 = xmid + dx / 2
    lon0, lat0 = crs.transform_point(x0, ymid, ax.projection)
    lon1, lat1 = crs.transform_point(x1, ymid, ax.projection)
    geod = Geod(ellps='WGS84')
    dr = geod.inv(lon0, lat0, lon1, lat1)[2] / unit
    dxdr = dx / dr

    # axes坐标系的位置转为data坐标.
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

def add_box(ax, extents, **kwargs):
    '''
    在Axes上添加一个方框.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    extents : 4-tuple of float
        方框范围[xmin, xmax, ymin, ymax].

    **kwargs
        创建Rectangle对象的关键字参数.
        例如linewidth, edgecolor, facecolor和transform等.

    Returns
    -------
    rect : Rectangle
        方框对象.
    '''
    xmin, xmax, ymin, ymax = extents
    dx = xmax - xmin
    dy = ymax - ymin
    rect = mpatches.Rectangle((xmin, ymin), dx, dy, **kwargs)
    ax.add_patch(rect)

    return rect

# TODO: inset_axes实现.
def move_axes_to_corner(ax, ref_ax, shrink=0.4, loc='bottom right'):
    '''
    讲ax等比例缩小并放置在ref_ax的角落位置.

    Parameters
    ----------
    ax : Axes
        目标Axes. 若为GeoAxes, 需要提前调用ax.set_extent.

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
    new_bbox = mtransforms.Bbox.from_extents(x0, y0, x1, y1)
    ax.set_position(new_bbox)

def add_side_axes(ax, loc, pad, depth):
    '''
    在原有的Axes旁边新添一个等高或等宽的Axes并返回该对象.

    Parameters
    ----------
    ax : Axes or array_like of Axes
        原有的Axes, 也可以是一组Axes构成的数组.

    loc : {'left', 'right', 'bottom', 'top'}
        新Axes相对于旧Axes的位置.

    pad : float
        新旧Axes的间距.

    depth : float
        新Axes的宽度或高度.

    Returns
    -------
    side_ax : Axes
        新Axes对象.
    '''
    # 获取一组Axes的位置.
    axs = np.atleast_1d(ax).ravel()
    bbox = mtransforms.Bbox.union([ax.get_position() for ax in axs])

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
    side_bbox = mtransforms.Bbox.from_extents(x0, y0, x1, y1)
    side_ax = axs[0].figure.add_axes(side_bbox)

    return side_ax

def get_slice_xticks(
    lon, lat, ntick=6, decimals=2,
    lon_formatter=None, lat_formatter=None
):
    '''
    返回垂直剖面图所需的横坐标, 刻度位置和刻度标签.

    用经纬度数组的点数表示横坐标, 在横坐标上取ntick个等距的刻度,
    利用线性插值计算每个刻度标签的经纬度值.

    Parameters
    ----------
    lon : (npt,) array_like
        剖面对应的经度数组.

    lat : (npt,) array_like
        剖面对应的纬度数组.

    ntick : int, optional
        刻度的数量. 默认为6.

    decimals : int, optional
        刻度标签的小数位数. 默认为2.

    lon_formatter : Formatter, optional
        刻度标签里经度的Formatter. 默认为None, 表示LongitudeFormatter.

    lat_formatter : Formatter, optional
        刻度标签里纬度的Formatter. 默认为None, 表示LatitudeFormatter.

    Returns
    -------
    x : (npt,) ndarray
        剖面数据的横坐标. 数值等于np.arange(npt).

    xticks : (ntick,) ndarray
        横坐标的刻度位置.

    xticklabels : (ntick,) list of str
        横坐标的刻度标签. 用刻度处的经纬度值表示.
    '''
    # 线性插值计算刻度的经纬度值.
    npt = len(lon)
    x = np.arange(npt)
    xticks = np.linspace(0, npt - 1, ntick)
    lon_ticks = np.interp(xticks, x, lon).round(decimals)
    lat_ticks = np.interp(xticks, x, lat).round(decimals)

    # 获取字符串形式的刻度标签.
    xticklabels = []
    if lon_formatter is None:
        lon_formatter = LongitudeFormatter()
    if lat_formatter is None:
        lat_formatter = LatitudeFormatter()
    for i in range(ntick):
        lon_label = lon_formatter(lon_ticks[i])
        lat_label = lat_formatter(lat_ticks[i])
        xticklabels.append(lon_label + '\n' + lat_label)

    return x, xticks, xticklabels

def make_qualitative_cmap(colors):
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
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=-0.5, vmax=N-0.5)
    ticks = np.arange(N)

    return cmap, norm, ticks

def get_aod_cmap():
    '''返回适用于AOD的cmap.'''
    filepath = Path(__file__).parent / 'data' / 'NEO_modis_aer_od.csv'
    rgb = np.loadtxt(str(filepath), delimiter=',') / 256
    cmap = mcolors.ListedColormap(rgb)

    return cmap

def letter_axes(axes, x, y, **kwargs):
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
            xi, yi, f'({letter})', ha='center', va='center',
            transform=ax.transAxes, **kwargs
        )

def make_gif(
    filepaths_img, filepath_gif,
    duration=500, loop=0, optimize=False
):
    '''
    制作GIF图.

    Parameters
    ----------
    filepaths_img : list of str or Path
        图片路径的列表. 要求至少含两个元素.

    filepath_gif : str or Path
        输出GIF图片的路径.

    duration : int or list or tuple, optional
        每一帧的持续时间, 以毫秒为单位. 也可以用列表或元组分别指定每一帧的持续时间.
        默认为500ms=0.5s.

    loop : int, optional
        GIF图片循环播放的次数. 默认无限循环.

    optimize : bool, optional
        尝试压缩GIF图片的调色板.
    '''
    if len(filepaths_img) < 2:
        raise ValueError('至少需要两张图片')

    images = [Image.open(str(filepath)) for filepath in filepaths_img]
    images[0].save(
        str(filepath_gif), save_all=True, append_images=images[1:],
        duration=duration, loop=loop, optimize=optimize
    )