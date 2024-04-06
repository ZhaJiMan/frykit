import math
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union
from weakref import WeakKeyDictionary, WeakValueDictionary

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry as sgeom
from cartopy.mpl.feature_artist import _GeomKey
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cbook import normalize_kwargs
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize
from matplotlib.contour import ContourSet
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as Path
from matplotlib.quiver import Quiver, QuiverKey
from matplotlib.text import Text
from matplotlib.ticker import Formatter
from matplotlib.transforms import Affine2D, Bbox, ScaledTranslation
from pyproj import Geod
from shapely.ops import unary_union

import frykit.shp as fshp
from frykit import DATA_DIRPATH
from frykit._artist import QuiverLegend

# 当polygon的引用计数为零时, 弱引用会自动清理缓存.
# cartopy是直接缓存Path, 但测试后发现差距不大.
_key_to_polygon = WeakValueDictionary()
_key_to_crs_to_transformed_polygon = WeakKeyDictionary()

_USE_FAST_TRANSFORM = True


def use_fast_transform(b: bool) -> None:
    '''
    是否启用快速变换.

    快速变换基于pyproj.Transformer, 速度更快但可能在地图边界产生错误的连线.
    普通变换基于cartopy.crs.Projection.project_geometry, 速度更慢但结果更正确.
    '''
    global _USE_FAST_TRANSFORM
    _USE_FAST_TRANSFORM = b
    _key_to_polygon.clear()
    _key_to_crs_to_transformed_polygon.clear()


def _cached_transform_polygons(
    polygons: Sequence[fshp.PolygonType], crs_from: ccrs.CRS, crs_to: ccrs.CRS
) -> list[fshp.PolygonType]:
    '''对一组多边形做坐标变换并缓存结果.'''
    if _USE_FAST_TRANSFORM:
        transform_polygon = fshp.GeometryTransformer(crs_from, crs_to)
    else:
        transform_polygon = lambda x: crs_to.project_geometry(x, crs_from)

    values = []
    for polygon in polygons:
        key = _GeomKey(polygon)
        _key_to_polygon.setdefault(key, polygon)
        mapping = _key_to_crs_to_transformed_polygon.setdefault(key, {})
        value = mapping.get(crs_to)
        if value is None:
            value = transform_polygon(polygon)
            mapping[crs_to] = value
        values.append(value)

    return values


def _cached_transform_polygon(
    polygon: fshp.PolygonType, crs_from: ccrs.CRS, crs_to: ccrs.CRS
) -> fshp.PolygonType:
    '''对一个多边形做坐标变换并缓存结果.'''
    return _cached_transform_polygons([polygon], crs_from, crs_to)[0]


def add_polygons(
    ax: Axes,
    polygons: Sequence[fshp.PolygonType],
    crs: Optional[ccrs.CRS] = None,
    **kwargs: Any,
) -> PathCollection:
    '''
    将一组多边形添加到Axes上.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    polygons : sequence of PolygonType
        多边形构成的序列.

    crs : CRS, optional
        当ax是GeoAxes时会将多边形从crs表示的坐标系变换到ax所在的坐标系上.
        默认为None, 表示PlateCarree().

    **kwargs
        PathCollection类的关键字参数.
        例如edgecolor, facecolor, cmap, norm和array等.
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        代表一组多边形的集合对象.
    '''
    array = kwargs.get('array', None)
    if array is not None and len(array) != len(polygons):
        raise ValueError('array的长度与polygons不匹配')

    # GeoAxes会对多边形做坐标变换.
    if not isinstance(ax, Axes):
        raise ValueError('ax不是Axes')
    elif isinstance(ax, GeoAxes):
        crs = ccrs.PlateCarree() if crs is None else crs
        polygons = _cached_transform_polygons(polygons, crs, ax.projection)
    else:
        if crs is not None:
            raise ValueError('ax不是GeoAxes时crs只能为None')

    # PathCollection比PathPatch更快.
    paths = [fshp.polygon_to_path(polygon) for polygon in polygons]
    kwargs.setdefault('transform', ax.transData)
    pc = PathCollection(paths, **kwargs)
    ax.add_collection(pc)
    ax._request_autoscale_view()

    return pc


def add_polygon(
    ax: Axes,
    polygon: fshp.PolygonType,
    crs: Optional[ccrs.CRS] = None,
    **kwargs: Any,
) -> PathCollection:
    '''
    将一个多边形添加到Axes上.

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
        PathCollection类的关键字参数.
        例如edgecolor, facecolor, cmap, norm和array等.
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        只含一个多边形的集合对象.
    '''
    return add_polygons(ax, [polygon], crs, **kwargs)


def _get_boundary(ax: GeoAxes) -> sgeom.Polygon:
    '''将GeoAxes.patch转为data坐标系下的多边形.'''
    patch = ax.patch
    patch._adjust_location()
    trans = patch.get_transform() - ax.transData
    path = patch.get_path().transformed(trans)
    boundary = sgeom.Polygon(path.vertices)

    return boundary


ArtistType = Union[Artist, Sequence[Artist]]


def clip_by_polygon(
    artist: ArtistType,
    polygon: fshp.PolygonType,
    crs: Optional[ccrs.CRS] = None,
    strict: bool = False,
) -> None:
    '''
    用多边形裁剪Artist, 只显示多边形内的内容.

    Parameters
    ----------
    artist : ArtistType
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

    strict : bool, optional
        是否使用更严格的裁剪方法. 默认为False.
        为True时即便GeoAxes的边界不是矩形也能避免出界.
    '''
    artists = []
    for a in artist if isinstance(artist, Sequence) else [artist]:
        if isinstance(a, ContourSet) and mpl.__version__ < '3.8.0':
            artists.extend(a.collections)
        else:
            artists.append(a)
    ax = artists[0].axes
    for i in range(1, len(artists)):
        if artists[i].axes is not ax:
            raise ValueError('一组Artist必须属于同一个Axes')

    if not isinstance(ax, Axes):
        raise ValueError('ax不是Axes')
    if isinstance(ax, GeoAxes):
        crs = ccrs.PlateCarree() if crs is None else crs
        polygon = _cached_transform_polygon(polygon, crs, ax.projection)
        # TODO: 通过装饰Artist.draw方法实现.
        if strict:  # 在data坐标系求polygon和ax.patch的交集.
            polygon &= _get_boundary(ax)
    else:
        # Axes会自动给Artist设置clipbox, 所以不会出界.
        if crs is not None:
            raise ValueError('ax不是GeoAxes时crs只能为None')
    path = fshp.polygon_to_path(polygon)
    trans = ax.transData

    # TODO:
    # 用字体位置来判断仍然会有出界的情况.
    # 用t.get_window_extent()的结果和polygon做运算.
    for a in artists:
        a.set_clip_on(True)
        a.set_clip_box(ax.bbox)
        if isinstance(a, Text):
            point = sgeom.Point(a.get_position())
            if not polygon.contains(point):
                a.set_visible(False)
        else:
            a.set_clip_path(path, trans)


def clip_by_polygons(
    artist: ArtistType,
    polygons: Sequence[fshp.PolygonType],
    crs: Optional[ccrs.CRS] = None,
    strict: bool = False,
) -> None:
    '''
    用一组多边形裁剪Artist, 只显示多边形内的内容.

    该函数不能像clip_by_polygon一样利用缓存加快二次运行的速度.

    Parameters
    ----------
    artist : ArtistType
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

    strict : bool, optional
        是否使用更严格的裁剪方法. 默认为False.
        为True时即便GeoAxes的边界不是矩形也能避免出界.
    '''
    polygon = unary_union(polygons)
    clip_by_polygon(artist, polygon, crs, strict)


# 缓存常用数据.
_data_cache = {}


def _get_cached_cn_border() -> fshp.PolygonType:
    '''获取缓存的中国国界.'''
    polygon = _data_cache.get('cn_border')
    if polygon is None:
        polygon = fshp.get_cn_border()
        _data_cache['cn_border'] = polygon

    return polygon


def _get_cached_nine_line() -> fshp.PolygonType:
    '''获取缓存的九段线.'''
    polygon = _data_cache.get('nine_line')
    if polygon is None:
        polygon = fshp.get_nine_line()
        _data_cache['nine_line'] = polygon

    return polygon


def _get_cached_cn_provinces(
    province: Optional[fshp.GetCNKeyword] = None,
) -> list[fshp.PolygonType]:
    '''获取缓存的中国省界.'''
    polygons = _data_cache.get('cn_province')
    if polygons is None:
        polygons = fshp.get_cn_province()
        _data_cache['cn_province'] = polygons
    polygons = [polygons[i] for i in fshp._get_pr_locs(province)]

    return polygons


def _get_cached_cn_cities(
    city: Optional[fshp.GetCNKeyword] = None,
    province: Optional[fshp.GetCNKeyword] = None,
) -> list[fshp.PolygonType]:
    '''获取缓存的中国市界.'''
    polygons = _data_cache.get('cn_city')
    if polygons is None:
        polygons = fshp.get_cn_city()
        _data_cache['cn_city'] = polygons
    polygons = [polygons[i] for i in fshp._get_ct_locs(city, province)]

    return polygons


def _get_cached_countries() -> list[fshp.PolygonType]:
    '''获取缓存的所有国家的国界.'''
    polygons = _data_cache.get('countries')
    if polygons is None:
        polygons = fshp.get_countries()
        _data_cache['countries'] = polygons

    return polygons


def _get_cached_land() -> fshp.PolygonType:
    '''获取缓存的陆地.'''
    polygon = _data_cache.get('land')
    if polygon is None:
        polygon = fshp.get_land()
        _data_cache['land'] = polygon

    return polygon


def _get_cached_ocean() -> fshp.PolygonType:
    '''获取缓存的海洋.'''
    polygon = _data_cache.get('ocean')
    if polygon is None:
        polygon = fshp.get_ocean()
        _data_cache['ocean'] = polygon

    return polygon


def _init_pc_kwargs(kwargs: dict) -> dict:
    '''初始化传给PathCollection的参数.'''
    kwargs = normalize_kwargs(kwargs, PathCollection)
    kwargs.setdefault('linewidth', 0.5)
    kwargs.setdefault('edgecolor', 'k')
    kwargs.setdefault('facecolor', 'none')
    kwargs.setdefault('zorder', 1.5)

    return kwargs


# 竖版中国标准地图的投影.
# http://gi.m.mnr.gov.cn/202103/t20210312_2617069.html
CN_AZIMUTHAL_EQUIDISTANT = ccrs.AzimuthalEquidistant(
    central_longitude=105, central_latitude=35
)
# 网络墨卡托投影.
WEB_MERCATOR = ccrs.Mercator.GOOGLE


def add_cn_border(ax: Axes, **kwargs: Any) -> PathCollection:
    '''
    将中国国界添加到Axes上.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    **kwargs
        PathCollection类的关键字参数.
        例如edgecolor, facecolor, cmap, norm和array等.
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示国界的集合对象.
    '''
    kwargs = _init_pc_kwargs(kwargs)
    polygon = _get_cached_cn_border()
    pc = add_polygon(ax, polygon, **kwargs)

    return pc


def add_nine_line(ax: Axes, **kwargs: Any) -> PathCollection:
    '''
    将九段线添加到Axes上.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    **kwargs
        PathCollection类的关键字参数.
        例如edgecolor, facecolor, cmap, norm和array等.
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示九段线的集合对象.
    '''
    kwargs = _init_pc_kwargs(kwargs)
    polygon = _get_cached_nine_line()
    pc = add_polygon(ax, polygon, **kwargs)

    return pc


def add_cn_province(
    ax: Axes, province: Optional[fshp.GetCNKeyword] = None, **kwargs: Any
) -> PathCollection:
    '''
    将中国省界添加到Axes上.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    province : GetCNKeyword, optional
        单个省名或一组省名. 默认为None, 表示添加所有省.

    **kwargs
        PathCollection类的关键字参数.
        例如edgecolor, facecolor, cmap, norm和array等.
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示省界的集合对象.
    '''
    kwargs = _init_pc_kwargs(kwargs)
    polygons = _get_cached_cn_provinces(province)
    pc = add_polygons(ax, polygons, **kwargs)

    return pc


def add_cn_city(
    ax: Axes,
    city: Optional[fshp.GetCNKeyword] = None,
    province: Optional[fshp.GetCNKeyword] = None,
    **kwargs: Any,
) -> PathCollection:
    '''
    将中国市界添加到Axes上.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    city : GetCNKeyword, optional
        单个市名或一组市名. 默认为None, 表示添加所有市.

    province : GetCNKeyword, optional
        单个省名或一组省名, 添加属于某个省的所有市.
        默认为None, 表示不使用省名添加.
        不能同时指定city和province.

    **kwargs
        PathCollection类的关键字参数.
        例如edgecolor, facecolor, cmap, norm和array等.
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示市界的集合对象.
    '''
    kwargs = _init_pc_kwargs(kwargs)
    polygons = _get_cached_cn_cities(city, province)
    pc = add_polygons(ax, polygons, **kwargs)

    return pc


def add_countries(ax: Axes, **kwargs: Any) -> PathCollection:
    '''
    将所有国家的国界添加到Axes上.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    **kwargs
        PathCollection类的关键字参数.
        例如edgecolor, facecolor, cmap, norm和array等.
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示国界的集合对象.
    '''
    kwargs = _init_pc_kwargs(kwargs)
    polygons = _get_cached_countries()
    pc = add_polygons(ax, polygons, **kwargs)

    return pc


def add_land(ax: Axes, **kwargs: Any) -> PathCollection:
    '''
    将陆地添加到Axes上.

    注意默认zorder为-1
    全球数据可能因为地图边界产生错误的结果.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    **kwargs
        PathCollection类的关键字参数.
        例如edgecolor, facecolor, cmap, norm和array等.
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示陆地的集合对象.
    '''
    kwargs = _init_pc_kwargs(kwargs)
    polygon = _get_cached_land()
    pc = add_polygon(ax, polygon, **kwargs)

    return pc


def add_ocean(ax: Axes, **kwargs: Any) -> PathCollection:
    '''
    将海洋添加到Axes上.

    注意默认zorder为-1
    全球数据可能因为地图边界产生错误的结果.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    **kwargs
        PathCollection类的关键字参数.
        例如edgecolor, facecolor, cmap, norm和array等.
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示海洋的集合对象.
    '''
    kwargs = _init_pc_kwargs(kwargs)
    polygon = _get_cached_ocean()
    pc = add_polygon(ax, polygon, **kwargs)

    return pc


def clip_by_cn_border(artist: ArtistType, strict: bool = False) -> None:
    '''
    用中国国界裁剪Artist.

    Parameters
    ----------
    artist : ArtistType
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    strict : bool, optional
        是否使用更严格的裁剪方法. 默认为False.
        为True时即便GeoAxes的边界不是矩形也能避免出界.
    '''
    polygon = _get_cached_cn_border()
    clip_by_polygon(artist, polygon, strict=strict)


def clip_by_cn_province(
    artist: ArtistType, province: str, strict: bool = False
) -> None:
    '''
    用中国省界裁剪Artist.

    Parameters
    ----------
    artist : ArtistType
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    province : str
        单个省名.

    strict : bool, optional
        是否使用更严格的裁剪方法. 默认为False.
        为True时即便GeoAxes的边界不是矩形也能避免出界.
    '''
    if not isinstance(province, str):
        raise ValueError('只支持单个省')
    polygon = _get_cached_cn_provinces(province)[0]
    clip_by_polygon(artist, polygon, strict=strict)


def clip_by_cn_city(
    artist: ArtistType, city: str = None, strict: bool = False
) -> None:
    '''
    用中国市界裁剪Artist.

    Parameters
    ----------
    artist : ArtistType
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    city : str
        单个市名.

    strict : bool, optional
        是否使用更严格的裁剪方法. 默认为False.
        为True时即便GeoAxes的边界不是矩形也能避免出界.
    '''
    if not isinstance(city, str):
        raise ValueError('只支持单个市')
    polygon = _get_cached_cn_cities(city)[0]
    clip_by_polygon(artist, polygon, strict=strict)


def clip_by_land(artist: ArtistType, strict: bool = False) -> None:
    '''
    用陆地边界裁剪Artist.

    全球数据可能因为地图边界产生错误的结果.

    Parameters
    ----------
    artist : ArtistType
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    strict : bool, optional
        是否使用更严格的裁剪方法. 默认为False.
        为True时即便GeoAxes的边界不是矩形也能避免出界.
    '''
    clip_by_polygon(artist, _get_cached_land(), strict=strict)


def clip_by_ocean(artist: ArtistType, strict: bool = False) -> None:
    '''
    用海洋边界裁剪Artist.

    全球数据可能因为地图边界产生错误的结果.

    Parameters
    ----------
    artist : ArtistType
        被裁剪的Artist对象. 可以返回自以下方法:
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    strict : bool, optional
        是否使用更严格的裁剪方法. 默认为False.
        为True时即便GeoAxes的边界不是矩形也能避免出界.
    '''
    clip_by_polygon(artist, _get_cached_ocean(), strict=strict)


def add_texts(
    ax: Axes, x: Any, y: Any, s: Sequence[str], **kwargs: Any
) -> list[Text]:
    '''
    在Axes上添加一组文本.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    x : (n,) array_like
        文本的横坐标数组.

    y : (n,) array_like
        文本的纵坐标数组.

    s : (n,) sequence of str
        一组字符串文本.

    **kwargs
        Axes.text方法的关键字参数.
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

    Returns
    -------
    texts : (n,) list of Text
        一组Text对象.
    '''
    if not len(x) == len(y) == len(s):
        raise ValueError('x, y和s长度必须相同')
    return [ax.text(xi, yi, si, **kwargs) for xi, yi, si in zip(x, y, s)]


def _add_cn_texts(ax: Axes, table: pd.DataFrame, **kwargs: Any) -> list[Text]:
    '''用省或市的table在Axes上添加一组文本.'''
    kwargs = normalize_kwargs(kwargs, Text)
    kwargs.setdefault('fontsize', 'x-small')
    kwargs.setdefault('horizontalalignment', 'center')
    kwargs.setdefault('verticalalignment', 'center')

    # 加载精简的思源黑体.
    filepath = DATA_DIRPATH / 'zh_font.otf'
    if kwargs.get('fontname') is None and kwargs.get('fontfamily') is None:
        kwargs.setdefault('fontproperties', filepath)

    is_geoaxes = isinstance(ax, GeoAxes)
    kwargs.setdefault('clip_on', True)
    kwargs.setdefault('clip_box', ax.bbox if is_geoaxes else None)
    kwargs.setdefault(
        'transform', ccrs.PlateCarree() if is_geoaxes else ax.transData
    )

    return add_texts(
        ax=ax,
        x=table.iloc[:, 0],
        y=table.iloc[:, 1],
        s=table.iloc[:, 2],
        **kwargs,
    )


def label_cn_province(
    ax: Axes,
    province: Optional[fshp.GetCNKeyword] = None,
    short: bool = True,
    **kwargs: Any,
) -> list[Text]:
    '''
    在Axes上标注中国省名.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    province : GetCNKeyword, optional
        单个省名或一组省名. 默认为None, 表示标注所有省.

    short : bool, optional
        是否使用缩短的省名. 默认为True.

    **kwargs
        Axes.text方法的关键字参数.
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

    Returns
    -------
    texts : list of Text
        表示省名的Text对象.
    '''
    locs = fshp._get_pr_locs(province)
    key = 'short_name' if short else 'pr_name'
    table = fshp.PR_TABLE[['lon', 'lat', key]].iloc[locs]
    texts = _add_cn_texts(ax, table, **kwargs)

    return texts


def label_cn_city(
    ax: Axes,
    city: Optional[fshp.GetCNKeyword] = None,
    province: Optional[fshp.GetCNKeyword] = None,
    short: bool = True,
    **kwargs: Any,
) -> list[Text]:
    '''
    在Axes上标注中国市名.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    city : GetCNKeyword, optional
        单个市名或一组市名. 默认为None, 表示标注所有市.

    province: GetCNKeyword, optional
        单个省名或一组省名, 标注属于某个省的所有市.
        默认为None, 表示不使用省名标注.
        不能同时指定name和province.

    short : bool, optional
        是否使用缩短的市名. 默认为True.

    **kwargs
        Axes.text方法的关键字参数.
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

    Returns
    -------
    texts : list of Text
        表示市名的Text对象.
    '''
    locs = fshp._get_ct_locs(city, province)
    key = 'short_name' if short else 'ct_name'
    table = fshp.CT_TABLE[['lon', 'lat', key]].iloc[locs]
    texts = _add_cn_texts(ax, table, **kwargs)

    return texts


def _set_axes_ticks(
    ax: Axes,
    extents: Any,
    major_xticks: np.ndarray,
    major_yticks: np.ndarray,
    minor_xticks: np.ndarray,
    minor_yticks: np.ndarray,
    xformatter: Formatter,
    yformatter: Formatter,
) -> None:
    '''设置PlateCarree投影的Axes的范围和刻度.'''
    ax.set_xticks(major_xticks, minor=False)
    ax.set_yticks(major_yticks, minor=False)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)

    if extents is None:
        extents = [-180, 180, -90, 90]
    x0, x1, y0, y1 = extents
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def _set_simple_geoaxes_ticks(
    ax: GeoAxes,
    extents: Any,
    major_xticks: np.ndarray,
    major_yticks: np.ndarray,
    minor_xticks: np.ndarray,
    minor_yticks: np.ndarray,
    xformatter: Formatter,
    yformatter: Formatter,
) -> None:
    '''设置简单投影的GeoAxes的范围和刻度.'''
    crs = ccrs.PlateCarree()
    ax.set_xticks(major_xticks, minor=False, crs=crs)
    ax.set_yticks(major_yticks, minor=False, crs=crs)
    ax.set_xticks(minor_xticks, minor=True, crs=crs)
    ax.set_yticks(minor_yticks, minor=True, crs=crs)
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)

    if extents is None:
        ax.set_global()
    else:
        ax.set_extent(extents, crs)


def _set_complex_geoaxes_ticks(
    ax: GeoAxes,
    extents: Any,
    major_xticks: np.ndarray,
    major_yticks: np.ndarray,
    minor_xticks: np.ndarray,
    minor_yticks: np.ndarray,
    xformatter: Formatter,
    yformatter: Formatter,
) -> None:
    '''设置复杂投影的GeoAxes的范围和刻度.'''
    # 将地图边框设置为矩形.
    crs = ccrs.PlateCarree()
    if extents is None:
        proj_type = str(type(ax.projection)).split("'")[1].split('.')[-1]
        raise ValueError(f'在{proj_type}投影里extents为None会产生错误的刻度')
    ax.set_extent(extents, crs)

    eps = 1
    npts = 100
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lon0, lon1, lat0, lat1 = ax.get_extent(crs)
    lon0 -= eps
    lon1 += eps
    lat0 -= eps
    lat1 += eps

    # 在data坐标系用LineString表示地图边框四条边长.
    lineB = sgeom.LineString([(x0, y0), (x1, y0)])
    lineT = sgeom.LineString([(x0, y1), (x1, y1)])
    lineL = sgeom.LineString([(x0, y0), (x0, y1)])
    lineR = sgeom.LineString([(x1, y0), (x1, y1)])

    def get_two_xticks(
        xticks: np.ndarray,
    ) -> tuple[list[float], list[float], list[str], list[str]]:
        '''获取地图上下边框的x刻度和刻度标签.'''
        xticksB = []
        xticksT = []
        xticklabelsB = []
        xticklabelsT = []
        lats = np.linspace(lat0, lat1, npts)
        xticks = xticks[(xticks >= lon0) & (xticks <= lon1)]
        for xtick in xticks:
            lons = np.full_like(lats, xtick)
            lon_line = sgeom.LineString(np.column_stack((lons, lats)))
            lon_line = ax.projection.project_geometry(lon_line, crs)
            pointB = lineB.intersection(lon_line)
            if isinstance(pointB, sgeom.Point) and not pointB.is_empty:
                xticksB.append(pointB.x)
                xticklabelsB.append(xformatter(xtick))
            pointT = lineT.intersection(lon_line)
            if isinstance(pointT, sgeom.Point) and not pointT.is_empty:
                xticksT.append(pointT.x)
                xticklabelsT.append(xformatter(xtick))

        return xticksB, xticksT, xticklabelsB, xticklabelsT

    def get_two_yticks(
        yticks: np.ndarray,
    ) -> tuple[list[float], list[float], list[str], list[str]]:
        '''获取地图左右边框的y刻度和刻度标签.'''
        yticksL = []
        yticksR = []
        yticklabelsL = []
        yticklabelsR = []
        lons = np.linspace(lon0, lon1, npts)
        yticks = yticks[(yticks >= lat0) & (yticks <= lat1)]
        for ytick in yticks:
            lats = np.full_like(lons, ytick)
            lat_line = sgeom.LineString(np.column_stack((lons, lats)))
            lat_line = ax.projection.project_geometry(lat_line, crs)
            pointL = lineL.intersection(lat_line)
            if isinstance(pointL, sgeom.Point) and not pointL.is_empty:
                yticksL.append(pointL.y)
                yticklabelsL.append(yformatter(ytick))
            pointR = lineR.intersection(lat_line)
            if isinstance(pointR, sgeom.Point) and not pointR.is_empty:
                yticksR.append(pointR.y)
                yticklabelsR.append(yformatter(ytick))

        return yticksL, yticksR, yticklabelsL, yticklabelsR

    # 通过隐藏部分刻度, 实现上下刻度不同的效果.
    major_xticksB, major_xticksT, major_xticklabelsB, major_xticklabelsT = (
        get_two_xticks(major_xticks)
    )
    minor_xticksB, minor_xticksT, _, _ = get_two_xticks(minor_xticks)
    ax.set_xticks(major_xticksB + major_xticksT, minor=False)
    ax.set_xticks(minor_xticksB + minor_xticksT, minor=True)
    ax.set_xticklabels(major_xticklabelsB + major_xticklabelsT, minor=False)
    major_numB = len(major_xticksB)
    for tick in ax.xaxis.get_major_ticks()[:major_numB]:
        tick.tick2line.set_alpha(0)
        tick.label2.set_alpha(0)
    for tick in ax.xaxis.get_major_ticks()[major_numB:]:
        tick.tick1line.set_alpha(0)
        tick.label1.set_alpha(0)
    minor_numB = len(minor_xticksB)
    for tick in ax.xaxis.get_minor_ticks()[:minor_numB]:
        tick.tick2line.set_alpha(0)
    for tick in ax.xaxis.get_minor_ticks()[minor_numB:]:
        tick.tick1line.set_alpha(0)

    # 通过隐藏部分刻度, 实现左右刻度不同的效果.
    major_yticksL, major_yticksR, major_yticklabelsL, major_yticklabelsR = (
        get_two_yticks(major_yticks)
    )
    minor_yticksL, minor_yticksR, _, _ = get_two_yticks(minor_yticks)
    ax.set_yticks(major_yticksL + major_yticksR, minor=False)
    ax.set_yticks(minor_yticksL + minor_yticksR, minor=True)
    ax.set_yticklabels(major_yticklabelsL + major_yticklabelsR, minor=False)
    major_numL = len(major_yticksL)
    for tick in ax.yaxis.get_major_ticks()[:major_numL]:
        tick.tick2line.set_alpha(0)
        tick.label2.set_alpha(0)
    for tick in ax.yaxis.get_major_ticks()[major_numL:]:
        tick.tick1line.set_alpha(0)
        tick.label1.set_alpha(0)
    minor_numL = len(minor_yticksL)
    for tick in ax.yaxis.get_minor_ticks()[:minor_numL]:
        tick.tick2line.set_alpha(0)
    for tick in ax.yaxis.get_minor_ticks()[minor_numL:]:
        tick.tick1line.set_alpha(0)


def _interp_minor_ticks(major_ticks: Any, m: int) -> np.ndarray:
    '''在主刻度的每段间隔内线性插值出m个次刻度.'''
    n = len(major_ticks)
    if n == 0 or m <= 0:
        return np.array([])

    L = n + (n - 1) * m
    x = np.array([i for i in range(L) if i % (m + 1) > 0]) / (L - 1)
    xp = np.linspace(0, 1, n)
    minor_ticks = np.interp(x, xp, major_ticks)

    return minor_ticks


def set_map_ticks(
    ax: Axes,
    extents: Optional[Any] = None,
    xticks: Optional[Any] = None,
    yticks: Optional[Any] = None,
    dx: float = 10,
    dy: float = 10,
    mx: int = 0,
    my: int = 0,
    xformatter: Optional[Formatter] = None,
    yformatter: Optional[Formatter] = None,
) -> None:
    '''
    设置地图的范围和刻度.

    当ax是普通Axes时, 认为其投影为PlateCarree().
    当ax是GeoAxes时, 如果设置效果有误, 建议换用GeoAxes.gridlines.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    extents : (4,) array_like, optional
        经纬度范围[lon0, lon1, lat0, lat1]. 默认为None, 表示显示全球.
        当GeoAxes的投影不是PlateCarree或Mercator时extents不能为None.

    xticks : array_like, optional
        x轴主刻度的坐标, 单位为经度. 默认为None, 表示使用dx参数.

    yticks : array_like, optional
        y轴主刻度的坐标, 单位为纬度. 默认为None, 表示使用dy参数.

    dx : float, optional
        以dx为间隔从-180度开始生成x轴主刻度的间隔. 默认为10度.
        xticks不为None时会覆盖该参数.

    dy : float, optional
        以dy为间隔从-90度开始生成y轴主刻度的间隔. 默认为10度.
        yticks不为None时会覆盖该参数.

    mx : int, optional
        经度主刻度之间次刻度的个数. 默认为0.

    my : int, optional
        纬度主刻度之间次刻度的个数. 默认为0.

    xformatter : Formatter, optional
        x轴刻度标签的Formatter. 默认为None, 表示LongitudeFormatter().

    yformatter : Formatter, optional
        y轴刻度标签的Formatter. 默认为None, 表示LatitudeFormatter().
    '''
    if xticks is None:
        major_xticks = np.arange(math.floor(360 / dx) + 1) * dx - 180
    else:
        major_xticks = np.asarray(xticks)

    if yticks is None:
        major_yticks = np.arange(math.floor(180 / dy) + 1) * dy - 90
    else:
        major_yticks = np.asarray(yticks)

    if not isinstance(mx, int) or mx < 0:
        raise ValueError('mx只能是非负整数')
    minor_xticks = _interp_minor_ticks(major_xticks, mx)

    if not isinstance(my, int) or my < 0:
        raise ValueError('my只能是非负整数')
    minor_yticks = _interp_minor_ticks(major_yticks, my)

    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()

    if not isinstance(ax, Axes):
        raise ValueError('ax不是Axes')
    elif isinstance(ax, GeoAxes):
        if isinstance(ax.projection, (ccrs.PlateCarree, ccrs.Mercator)):
            setter = _set_simple_geoaxes_ticks
        else:
            setter = _set_complex_geoaxes_ticks
    else:
        setter = _set_axes_ticks

    setter(
        ax,
        extents,
        major_xticks,
        major_yticks,
        minor_xticks,
        minor_yticks,
        xformatter,
        yformatter,
    )


def add_quiver_legend(
    Q: Quiver,
    U: float,
    units: str = 'm/s',
    width: float = 0.15,
    height: float = 0.15,
    loc: Literal[
        'bottom left', 'bottom right', 'top left', 'top right'
    ] = 'bottom right',
    quiver_key_kwargs: Optional[dict] = None,
    patch_kwargs: Optional[dict] = None,
) -> tuple[Rectangle, QuiverKey]:
    '''
    为Quiver添加图例.

    图例由背景方框patch和风箭头key组成.
    key下方有形如'{U} {units}'的标签.

    Parameters
    ----------
    Q : Quiver
        Axes.quiver返回的对象.

    U : float
        箭头长度.

    units : str, optional
        标签单位. 默认为m/s.

    width : float, optional
        图例宽度. 基于Axes坐标, 默认为0.15

    height : float, optional
        图例高度. 基于Axes坐标, 默认为0.15

    loc : {'bottom left', 'bottom right', 'top left', 'top right'}, optional
        图例位置. 默认为'bottom right'.

    quiver_key_kwargs : dict, optional
        QuiverKey类的关键字参数.
        例如labelsep, labelcolor, fontproperties等.
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiverkey.html

    patch_kwargs : dict, optional
        表示背景方框的Retangle类的关键字参数.
        例如linewidth, edgecolor, facecolor等.
        https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html

    Returns
    -------
    quiver_legend : QuiverLegend
        图例对象.
    '''
    quiver_legend = QuiverLegend(
        Q, U, units, width, height, loc, quiver_key_kwargs, patch_kwargs
    )
    Q.axes.add_artist(quiver_legend)

    return quiver_legend


def add_compass(
    ax: Axes,
    x: float,
    y: float,
    angle: Optional[float] = None,
    size: float = 20,
    style: Literal['arrow', 'star', 'circle'] = 'arrow',
    path_kwargs: Optional[dict] = None,
    text_kwargs: Optional[dict] = None,
) -> tuple[PathCollection, Text]:
    '''
    向Axes添加指北针.

    调用函数前需要先固定GeoAxes的显示范围, 否则可能出现错误的结果.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    x, y : float
        指北针的横纵坐标. 基于axes坐标系.

    angle : float, optional
        指北针的方向, 从x轴逆时针方向算起, 单位为度. 默认为None.
        当ax是GeoAxes时默认自动计算角度, 否则默认表示90度.

    size : float, optional
        指北针的大小, 单位为点(point). 默认为20.

    style : {'arrow', 'star', 'circle'}, optional
        指北针的造型. 默认为'arrow'.

    path_kwargs : dict, optional
        表示指北针的PathCollection类的关键字参数.
        例如linewidth, edgecolor, facecolor等.

    text_kwargs : dict, optional
        绘制指北针N字时Axes.Text方法的关键字参数.
        默认为None, 表示使用默认参数.

    Returns
    -------
    pc : PathCollection
        表示指北针的Collection对象.

    t : Text
        指北针N字对象.
    '''
    # 设置箭头参数.
    path_kwargs = normalize_kwargs(path_kwargs, PathCollection)
    path_kwargs.setdefault('linewidth', 1)
    path_kwargs.setdefault('edgecolor', 'k')
    if style == 'circle':
        path_kwargs.setdefault('facecolor', ['none', 'k', 'w'])
    else:
        path_kwargs.setdefault('facecolor', ['k', 'w'])
    path_kwargs.setdefault('zorder', 3)
    path_kwargs.setdefault('clip_on', False)

    # 设置文字参数.
    text_kwargs = normalize_kwargs(text_kwargs, Text)
    text_kwargs.setdefault('fontsize', size / 1.5)

    # 计算(lon, lat)到(lon, lat + 1)的角度.
    # 当(x, y)超出Axes范围时会计算出无意义的角度.
    if angle is None:
        if isinstance(ax, GeoAxes):
            crs = ccrs.PlateCarree()
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
    elif style == 'circle':
        width = axis = head * 2 / 3
        radius = head * 2 / 5
        theta = np.linspace(0, 2 * np.pi, 100)
        rx = radius * np.cos(theta) + head / 9
        ry = radius * np.sin(theta)
        verts1 = np.column_stack((rx, ry))
        verts2 = [(0, 0), (axis, 0), (axis - head, width / 2), (0, 0)]
        verts3 = [(0, 0), (axis - head, -width / 2), (axis, 0), (0, 0)]
        paths = [Path(verts1), Path(verts2), Path(verts3)]
    else:
        raise ValueError('style参数错误')

    # 添加指北针.
    pc = PathCollection(paths, transform=trans, **path_kwargs)
    ax.add_collection(pc)

    # 添加N字.
    pad = head / 3
    t = ax.text(
        x=axis + pad,
        y=0,
        s='N',
        ha='center',
        va='center',
        rotation=angle - 90,
        transform=trans,
        **text_kwargs,
    )

    return pc, t


def add_map_scale(
    ax: Axes,
    x: float,
    y: float,
    length: float = 1000,
    units: Literal['m', 'km'] = 'km',
) -> Axes:
    '''
    向Axes添加地图比例尺.

    当ax是普通Axes时, 认为其投影为PlateCarree(), 然后计算比例尺长度.
    当ax是GeoAxes时, 根据ax.projection计算比例尺长度.
    调用函数前需要先固定GeoAxes的显示范围, 否则可能出现错误的结果.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    x, y : float
        比例尺左端的横纵坐标. 基于axes坐标系.

    length : float, optional
        比例尺的长度. 默认为1000.

    units : {'m', 'km'}, optional
        比例尺长度的单位. 默认为'km'.

    Returns
    -------
    map_scale : Axes
        表示比例尺的Axes对象. 刻度可以直接通过scale.set_xticks进行修改.
    '''
    if units == 'km':
        unit = 1000
    elif units == 'm':
        unit = 1
    else:
        raise ValueError('units参数错误')

    if isinstance(ax, GeoAxes):
        # 取地图中心一小段水平线计算单位投影坐标的长度.
        crs = ccrs.PlateCarree()
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
    else:
        # 取地图中心的纬度计算单位经度的长度.
        Re = 6371e3
        L = 2 * math.pi * Re / 360
        lat0, lat1 = ax.get_ylim()
        lat = (lat0 + lat1) / 2
        drdx = L * math.cos(math.radians(lat))
        dxdr = unit / drdx

    # axes坐标转为data坐标.
    axes_to_data = ax.transAxes - ax.transData
    x, y = axes_to_data.transform((x, y))
    width = length * dxdr

    # 避免全局的rc设置影响刻度的样式.
    bounds = [x, y, width, 1e-4 * width]
    with plt.style.context('default'):
        map_scale = ax.inset_axes(bounds, transform=ax.transData)
    map_scale._map_scale = None  # 标识符.
    map_scale.tick_params(
        which='both',
        left=False,
        labelleft=False,
        bottom=False,
        labelbottom=False,
        top=True,
        labeltop=True,
        labelsize='small',
    )
    map_scale.set_xlabel(units, fontsize='medium')
    map_scale.set_xlim(0, length)

    return map_scale


def _path_from_extents(x0, x1, y0, y1, ccw=True) -> Path:
    '''根据方框范围构造Path对象. ccw表示逆时针.'''
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    if not ccw:
        verts.reverse()
    path = Path(verts)

    return path


# TODO: 复杂投影.
def gmt_style_frame(ax: Axes, width: float = 5, **kwargs: Any) -> None:
    '''
    为Axes设置GMT风格的边框.

    调用函数前需要先固定Axes的显示范围和刻度, 否则可能出现错误的结果.

    Parameters
    ----------
    ax : Axes
        目标Axes. 当ax是add_map_scale的返回值时只设置上边框.

    width : float, optional
        边框的宽度. 单位为点(point), 默认为5.

    **kwargs
        PathCollection类的关键字参数.
        例如linewidth, edgecolor, facecolor等.
        https://matplotlib.org/stable/api/collections_api.html
    '''
    is_geoaxes = isinstance(ax, GeoAxes)
    if is_geoaxes and not isinstance(
        ax.projection, (ccrs.PlateCarree, ccrs.Mercator)
    ):
        raise ValueError('只支持PlateCarree和Mercator投影')

    # 设置条纹参数.
    kwargs = normalize_kwargs(kwargs, PathCollection)
    kwargs.setdefault('linewidth', 1)
    kwargs.setdefault('edgecolor', 'k')
    kwargs.setdefault('facecolor', ['k', 'w'])
    kwargs.setdefault('zorder', 3)
    kwargs.setdefault('clip_on', False)

    # width单位转为英寸.
    if not hasattr(ax, '_map_scale'):
        ax.tick_params(
            which='both', left=True, right=True, top=True, bottom=True
        )
    ax.tick_params(which='major', length=width + 3.5)
    ax.tick_params(which='minor', length=0)
    width = width / 72

    # 确定物理坐标系和axes坐标系在xy方向上的缩放值.
    if is_geoaxes:
        ax.apply_aspect()  # 确定GeoAxes的transAxes.
    inches_to_axes = ax.figure.dpi_scale_trans - ax.transAxes
    matrix = inches_to_axes.get_matrix()
    dx = width * matrix[0, 0]
    dy = width * matrix[1, 1]

    # 条纹的transform: transData + transAxes
    xtrans = ax.get_xaxis_transform()
    ytrans = ax.get_yaxis_transform()

    # 收集[xmin, xmax]范围内所有刻度, 去重并排序.
    xticks = np.concatenate(
        (ax.xaxis.get_majorticklocs(), ax.xaxis.get_minorticklocs())
    )
    yticks = np.concatenate(
        (ax.yaxis.get_majorticklocs(), ax.yaxis.get_minorticklocs())
    )
    xmin, xmax = sorted(ax.get_xlim())
    ymin, ymax = sorted(ax.get_ylim())
    # xmin到xticks[0]之间也要填充条纹.
    xticks = np.append(xticks, (xmin, xmax))
    yticks = np.append(yticks, (ymin, ymax))
    xticks = xticks[(xticks >= xmin) & (xticks <= xmax)]
    yticks = yticks[(yticks >= ymin) & (yticks <= ymax)]
    # 通过round抵消GeoAxes投影变换的误差.
    if is_geoaxes:
        decimals = 6
        xticks = xticks.round(decimals)
        yticks = yticks.round(decimals)
    xticks = np.unique(xticks)
    yticks = np.unique(yticks)
    nx = len(xticks)
    ny = len(yticks)

    # 条纹从xmin开始黑白相间排列.
    top_paths = []
    for i in range(nx - 1):
        path = _path_from_extents(xticks[i], xticks[i + 1], 1, 1 + dy)
        top_paths.append(path)
    # axis倒转不影响条纹颜色顺序.
    if ax.xaxis.get_inverted():
        top_paths.reverse()
    top_pc = PathCollection(top_paths, transform=xtrans, **kwargs)
    ax.add_collection(top_pc)

    # 地图比例尺只画上边框.
    if hasattr(ax, '_map_scale'):
        return None

    bottom_paths = []
    for i in range(nx - 1):
        path = _path_from_extents(xticks[i], xticks[i + 1], -dy, 0)
        bottom_paths.append(path)
    if ax.xaxis.get_inverted():
        bottom_paths.reverse()
    bottom_pc = PathCollection(bottom_paths, transform=xtrans, **kwargs)
    ax.add_collection(bottom_pc)

    left_paths = []
    for i in range(ny - 1):
        path = _path_from_extents(-dx, 0, yticks[i], yticks[i + 1])
        left_paths.append(path)
    if ax.yaxis.get_inverted():
        left_paths.reverse()
    left_pc = PathCollection(left_paths, transform=ytrans, **kwargs)
    ax.add_collection(left_pc)

    right_paths = []
    for i in range(ny - 1):
        path = _path_from_extents(1, 1 + dx, yticks[i], yticks[i + 1])
        right_paths.append(path)
    if ax.yaxis.get_inverted():
        right_paths.reverse()
    right_pc = PathCollection(right_paths, transform=ytrans, **kwargs)
    ax.add_collection(right_pc)

    # 四个角落的方块单独用白色画出.
    corner_fc = top_pc.get_facecolor()[-1]
    for kw in ['fc', 'facecolor', 'facecolors']:
        if kw in kwargs:
            break
    kwargs[kw] = corner_fc
    corner_paths = [
        _path_from_extents(-dx, 0, -dy, 0),
        _path_from_extents(1, 1 + dx, -dy, 0),
        _path_from_extents(-dx, 0, 1, 1 + dy),
        _path_from_extents(1, 1 + dx, 1, 1 + dy),
    ]
    corner_pc = PathCollection(corner_paths, transform=ax.transAxes, **kwargs)
    ax.add_collection(corner_pc)


def add_box(
    ax: Axes, extents: Any, steps: int = 100, **kwargs: Any
) -> PathPatch:
    '''
    在Axes上添加一个方框.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    extents : (4,) array_like
        方框范围[x0, x1, y0, y1].

    steps: int, optional
        在方框上重采样出N*steps个点. 默认为 100.
        当ax是GeoAxes且指定transform关键字时能保证方框的平滑.

    **kwargs
        PathPatch类的关键字参数.
        例如linewidth, edgecolor, facecolor和transform等.
        https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.PathPatch.html

    Returns
    -------
    patch : PathPatch
        方框对象.
    '''
    # 设置参数.
    kwargs = normalize_kwargs(kwargs, PathPatch)
    kwargs.setdefault('edgecolor', 'r')
    kwargs.setdefault('facecolor', 'none')

    # 添加Patch.
    path = _path_from_extents(*extents).interpolated(steps)
    patch = PathPatch(path, **kwargs)
    ax.add_patch(patch)

    return patch


def load_test_data():
    '''读取测试用的数据. 包含地表2m气温(K)和水平10m风速.'''
    filepath = DATA_DIRPATH / 'test.npz'
    return np.load(str(filepath))


# TODO: inset_axes实现.
def move_axes_to_corner(
    ax: Axes,
    ref_ax: Axes,
    shrink: float = 0.4,
    loc: Literal[
        'bottom left', 'bottom right', 'top left', 'top right'
    ] = 'bottom right',
) -> None:
    '''
    讲ax等比例缩小并放置在ref_ax的角落位置.

    调用函数前需要先固定GeoAxes的显示范围, 否则可能出现错误的结果.

    Parameters
    ----------
    ax : Axes
        目标Axes.

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


# TODO: mpl_toolkits.axes_grid1实现.
def add_side_axes(
    ax: Any,
    loc: Literal['left', 'right', 'bottom', 'top'],
    pad: float,
    depth: float,
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
    nticks: int = 6,
    lon_formatter: Optional[Formatter] = None,
    lat_formatter: Optional[Formatter] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    '''
    返回垂直截面图所需的横坐标, 刻度位置和刻度标签.

    用经纬度的欧式距离表示横坐标, 在横坐标上取nticks个等距的刻度,
    利用线性插值计算每个刻度对应的经纬度值并用作刻度标签.

    Parameters
    ----------
    lon : (npts,) array_like
        横截面对应的经度数组.

    lat : (npts,) array_like
        横截面对应的纬度数组.

    nticks : int, optional
        刻度的数量. 默认为6.

    lon_formatter : Formatter, optional
        刻度标签里经度的Formatter, 用来控制字符串的格式.
        默认为None, 表示LongitudeFormatter.

    lat_formatter : Formatter, optional
        刻度标签里纬度的Formatter. 用来控制字符串的格式.
        默认为None, 表示LatitudeFormatter.

    Returns
    -------
    x : (npts,) ndarray
        横截面的横坐标.

    xticks : (nticks,) ndarray
        刻度的横坐标.

    xticklabels : (nticks,) list of str
        用经纬度表示的刻度标签.
    '''
    # 线性插值计算刻度的经纬度值.
    npts = len(lon)
    if npts <= 1:
        raise ValueError('lon和lat至少有2个元素')
    dlon = lon - lon[0]
    dlat = lat - lat[0]
    x = np.hypot(dlon, dlat)
    xticks = np.linspace(x[0], x[-1], nticks)
    tlon = np.interp(xticks, x, lon)
    tlat = np.interp(xticks, x, lat)

    # 获取字符串形式的刻度标签.
    xticklabels = []
    if lon_formatter is None:
        lon_formatter = LongitudeFormatter(number_format='.1f')
    if lat_formatter is None:
        lat_formatter = LatitudeFormatter(number_format='.1f')
    for i in range(nticks):
        lon_label = lon_formatter(tlon[i])
        lat_label = lat_formatter(tlat[i])
        xticklabels.append(lon_label + '\n' + lat_label)

    return x, xticks, xticklabels


def get_qualitative_palette(
    colors: Any,
) -> tuple[ListedColormap, Normalize, np.ndarray]:
    '''
    创建一组定性的colormap和norm, 同时返回刻度位置.

    Parameters
    ----------
    colors : (N,) sequence or (N, 3) or (N, 4) array_like
        colormap所含的颜色. 可以为含有颜色的序列或RGB(A)数组.

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
    norm = Normalize(vmin=-0.5, vmax=N - 0.5)
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
        self, boundaries: Any, vcenter: float = 0, clip: bool = False
    ) -> None:
        super().__init__(boundaries, len(boundaries) - 1, clip)
        boundaries = np.asarray(boundaries)
        self.N1 = np.count_nonzero(boundaries < vcenter)
        self.N2 = np.count_nonzero(boundaries > vcenter)
        if self.N1 < 1 or self.N2 < 1:
            raise ValueError('vcenter两侧至少各有一条边界')

    def __call__(
        self, value: Any, clip: Optional[bool] = None
    ) -> np.ma.MaskedArray:
        # 将BoundaryNorm的[0, N-1]又映射到[0.0, 1.0]内.
        result = super().__call__(value, clip)
        if self.N1 + self.N2 == self.N - 1:
            result = np.ma.where(
                result < self.N1,
                result / (2 * self.N1),
                (result - self.N1 + self.N2 + 1) / (2 * self.N2),
            )
        else:
            # 当result是MaskedArray时除以零不会报错.
            result = np.ma.where(
                result < self.N1,
                result / (2 * (self.N1 - 1)),
                (result - self.N1 + self.N2) / (2 * (self.N2 - 1)),
            )

        return result


def plot_colormap(
    cmap: Colormap,
    norm: Optional[Normalize] = None,
    extend: Optional[Literal['neither', 'both', 'min', 'max']] = None,
    ax: Optional[Axes] = None,
) -> Colorbar:
    '''快速展示一条colormap.'''
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 1))
        fig.subplots_adjust(bottom=0.5)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(
        mappable, cax=ax, orientation='horizontal', extend=extend
    )

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
        调用Axes.text时的关键字参数.
        例如fontsize, fontfamily和color等.
    '''
    axes = np.atleast_1d(axes)
    x = np.full_like(axes, x) if np.isscalar(x) else np.asarray(x)
    y = np.full_like(axes, y) if np.isscalar(y) else np.asarray(y)
    for i, (ax, xi, yi) in enumerate(zip(axes.flat, x.flat, y.flat)):
        letter = chr(97 + i)
        ax.text(
            x=xi,
            y=yi,
            s=f'({letter})',
            ha='center',
            va='center',
            transform=ax.transAxes,
            **kwargs,
        )
