import math
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union
from weakref import WeakKeyDictionary, WeakValueDictionary

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sgeom
from cartopy.crs import CRS, AzimuthalEquidistant, Mercator, PlateCarree
from cartopy.mpl.feature_artist import _GeomKey
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.cbook import normalize_kwargs
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as Path
from matplotlib.quiver import Quiver
from matplotlib.text import Text
from matplotlib.ticker import Formatter
from matplotlib.transforms import Bbox
from numpy.distutils.misc_util import is_sequence
from numpy.lib.npyio import NpzFile

import frykit.shp as fshp
from frykit import DATA_DIRPATH
from frykit._artist import *
from frykit._typing import StrOrSeq
from frykit.help import deprecator

# polygon 的引用计数为零时弱引用会自动清理缓存
_key_to_polygon = WeakValueDictionary()
_key_to_path = WeakKeyDictionary()
_key_to_crs_to_transformed_path = WeakKeyDictionary()


def _polygons_to_paths(polygons: Sequence[fshp.PolygonType]) -> list[Path]:
    '''将一组多边形转为 Path 并缓存结果'''
    paths = []
    for polygon in polygons:
        key = _GeomKey(polygon)
        _key_to_polygon.setdefault(key, polygon)
        path = _key_to_path.get(key)
        if path is None:
            path = fshp.polygon_to_path(polygon)
            _key_to_path[key] = path
        paths.append(path)

    return paths


def _transform_polygons_to_paths(
    polygons: Sequence[fshp.PolygonType],
    crs_from: CRS,
    crs_to: CRS,
    use_pyproj: bool = True,
) -> list[Path]:
    '''对一组多边形做坐标变换再转为 Path，并缓存结果。'''
    if use_pyproj:
        transform = fshp.GeometryTransformer(crs_from, crs_to)
    else:
        transform = lambda x: crs_to.project_geometry(x, crs_from)

    paths = []
    for polygon in polygons:
        key = _GeomKey(polygon)
        _key_to_polygon.setdefault(key, polygon)
        mapping = _key_to_crs_to_transformed_path.setdefault(key, {})
        value = mapping.get(crs_to)
        if value is None or value[0] != use_pyproj:
            polygon = transform(polygon)
            path = fshp.polygon_to_path(polygon)
            mapping[crs_to] = (use_pyproj, path)
        else:
            path = value[1]
        paths.append(path)

    return paths


def add_polygons(
    ax: Axes,
    polygons: Union[fshp.PolygonType, Sequence[fshp.PolygonType]],
    crs: Optional[CRS] = None,
    use_pyproj: bool = True,
    **kwargs: Any,
) -> PathCollection:
    '''
    将多边形添加到 Axes 上

    Parameters
    ----------
    ax : Axes
        目标 Axes

    polygons : PolygonType or sequence of PolygonType
        一个或一组多边形

    crs : CRS, optional
        当 ax 是 GeoAxes 时会将多边形从 crs 表示的坐标系变换到 ax 所在的坐标系上。
        默认为 None，表示 PlateCarree()。

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示多边形的集合对象
    '''
    if not is_sequence(polygons):
        polygons = [polygons]
    array = kwargs.get('array', None)
    if array is not None and len(array) != len(polygons):
        raise ValueError('array 的长度与 polygons 不匹配')

    if not isinstance(ax, Axes):
        raise ValueError('ax 不是 Axes')
    elif isinstance(ax, GeoAxes):
        if crs is None:
            crs = PlateCarree()
        paths = _transform_polygons_to_paths(
            polygons=polygons,
            crs_from=crs,
            crs_to=ax.projection,
            use_pyproj=use_pyproj,
        )
    else:
        if crs is not None:
            raise ValueError('ax 不是 GeoAxes 时 crs 只能为 None')
        paths = _polygons_to_paths(polygons)

    kwargs.setdefault('transform', ax.transData)
    pc = PathCollection(paths, **kwargs)
    ax.add_collection(pc)
    ax._request_autoscale_view()

    return pc


def _get_boundary(ax: GeoAxes) -> sgeom.Polygon:
    '''将 GeoAxes.patch 转为 data 坐标系里的多边形'''
    patch = ax.patch
    patch._adjust_location()
    trans = patch.get_transform() - ax.transData
    path = patch.get_path().transformed(trans)
    boundary = sgeom.Polygon(path.vertices)

    return boundary


ArtistOrSeq = Union[Artist, Sequence[Artist]]


def clip_by_polygon(
    artist: ArtistOrSeq,
    polygon: fshp.PolygonType,
    crs: Optional[CRS] = None,
    use_pyproj: bool = True,
    strict: bool = False,
) -> None:
    '''
    用多边形裁剪 Artist，只显示多边形内的内容。

    Parameters
    ----------
    artist : ArtistOrSeq
        被裁剪的 Artist对象。可以返回自以下方法：
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    polygon : PolygonType
        用于裁剪的多边形对象

    crs : CRS, optional
        当 Artist 在 GeoAxes 里时会将多边形从 crs 表示的坐标系变换到 Artist
        所在的坐标系上。默认为 None，表示 PlateCarree()。

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    strict : bool, optional
        是否使用更严格的裁剪方法。默认为 False。
        为 True 时即便 GeoAxes 的边界不是矩形也能避免出界。
    '''
    artists = []
    for a in artist if is_sequence(artist) else [artist]:
        if isinstance(a, ContourSet) and mpl.__version__ < '3.8.0':
            artists.extend(a.collections)
        else:
            artists.append(a)

    ax = artists[0].axes
    for i in range(1, len(artists)):
        if artists[i].axes is not ax:
            raise ValueError('一组 Artist 必须属于同一个 Axes')

    if not isinstance(ax, Axes):
        raise ValueError('ax 不是 Axes')
    is_geoaxes = isinstance(ax, GeoAxes)
    if is_geoaxes:
        if crs is None:
            crs = PlateCarree()
        path = _transform_polygons_to_paths(
            polygons=[polygon],
            crs_from=crs,
            crs_to=ax.projection,
            use_pyproj=use_pyproj,
        )[0]
        if strict:
            polygon = fshp.path_to_polygon(path)
            boudnary = _get_boundary(ax)
            path = fshp.polygon_to_path(polygon & boudnary)
    else:
        if crs is not None:
            raise ValueError('ax 不是 GeoAxes 时 crs 只能为 None')
        path = _polygons_to_paths([polygon])[0]

    # TODO:
    # 用字体位置来判断仍然会有出界的情况
    # 用 t.get_window_extent() 的结果和 polygon 做运算
    for a in artists:
        a.set_clip_on(True)
        if is_geoaxes:
            a.set_clip_box(ax.bbox)
        if isinstance(a, Text):
            point = sgeom.Point(a.get_position())
            if not polygon.contains(point):
                a.set_visible(False)
        else:
            a.set_clip_path(path, ax.transData)


def _init_pc_kwargs(kwargs: dict) -> dict:
    '''初始化传给 PathCollection 的参数'''
    kwargs = normalize_kwargs(kwargs, PathCollection)
    kwargs.setdefault('linewidth', 0.5)
    kwargs.setdefault('edgecolor', 'k')
    kwargs.setdefault('facecolor', 'none')
    kwargs.setdefault('zorder', 1.5)

    return kwargs


'''
竖版中国标准地图的投影
http://gi.m.mnr.gov.cn/202103/t20210312_2617069.html
'''
CN_AZIMUTHAL_EQUIDISTANT = AzimuthalEquidistant(
    central_longitude=105, central_latitude=35
)

# 网络墨卡托投影
WEB_MERCATOR = Mercator.GOOGLE


def add_cn_border(
    ax: Axes, use_pyproj: bool = True, **kwargs: Any
) -> PathCollection:
    '''
    在 Axes 上添加中国国界

    Parameters
    ----------
    ax : Axes
        目标 Axes

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示国界的集合对象
    '''
    return add_polygons(
        ax=ax,
        polygons=fshp.get_cn_border(),
        use_pyproj=use_pyproj,
        **_init_pc_kwargs(kwargs),
    )


def add_nine_line(
    ax: Axes, use_pyproj: bool = True, **kwargs: Any
) -> PathCollection:
    '''
    在 Axes 上添加九段线

    Parameters
    ----------
    ax : Axes
        目标 Axes

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示九段线的集合对象
    '''
    return add_polygons(
        ax=ax,
        polygons=fshp.get_nine_line(),
        use_pyproj=use_pyproj,
        **_init_pc_kwargs(kwargs),
    )


def add_cn_province(
    ax: Axes,
    province: Optional[StrOrSeq] = None,
    use_pyproj: bool = True,
    **kwargs: Any,
) -> PathCollection:
    '''
    在 Axes 上添加中国省界

    Parameters
    ----------
    ax : Axes
        目标 Axes

    province : StrOrSeq, optional
        单个或一组省名。默认为 None，表示获取所有省。

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示省界的集合对象
    '''
    return add_polygons(
        ax=ax,
        polygons=fshp.get_cn_province(province),
        use_pyproj=use_pyproj,
        **_init_pc_kwargs(kwargs),
    )


def add_cn_city(
    ax: Axes,
    city: Optional[StrOrSeq] = None,
    province: Optional[StrOrSeq] = None,
    use_pyproj: bool = True,
    **kwargs: Any,
) -> PathCollection:
    '''
    在 Axes 上添加中国市界

    Parameters
    ----------
    ax : Axes
        目标 Axes

    city : StrOrSeq, optional
        单个或一组市名。默认为 None，表示获取所有市。

    province : StrOrSeq, optional
        单个或一组省名，获取属于某个省的所有市。
        默认为 None，表示不指定省名。
        不能同时指定 city 和 province。

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示市界的集合对象
    '''
    return add_polygons(
        ax=ax,
        polygons=fshp.get_cn_city(city, province),
        use_pyproj=use_pyproj,
        **_init_pc_kwargs(kwargs),
    )


def add_countries(
    ax: Axes, use_pyproj: bool = True, **kwargs: Any
) -> PathCollection:
    '''
    在 Axes 上添加所有国家的国界

    注意全球数据可能在地图边界出现错误的结果。

    Parameters
    ----------
    ax : Axes
        目标 Axes

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示国界的集合对象
    '''
    return add_polygons(
        ax=ax,
        polygons=fshp.get_countries(),
        use_pyproj=use_pyproj,
        **_init_pc_kwargs(kwargs),
    )


def add_land(
    ax: Axes, use_pyproj: bool = True, **kwargs: Any
) -> PathCollection:
    '''
    在 Axes 上添加陆地

    注意全球数据可能在地图边界出现错误的结果。

    Parameters
    ----------
    ax : Axes
        目标 Axes

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示陆地的集合对象
    '''
    return add_polygons(
        ax=ax,
        polygons=fshp.get_land(),
        use_pyproj=use_pyproj,
        **_init_pc_kwargs(kwargs),
    )


def add_ocean(
    ax: Axes, use_pyproj: bool = True, **kwargs: Any
) -> PathCollection:
    '''
    在 Axes 上添加海洋

    注意全球数据可能在地图边界出现错误的结果。

    Parameters
    ----------
    ax : Axes
        目标 Axes

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    pc : PathCollection
        表示海洋的集合对象
    '''
    return add_polygons(
        ax=ax,
        polygons=fshp.get_ocean(),
        use_pyproj=use_pyproj,
        **_init_pc_kwargs(kwargs),
    )


def clip_by_cn_border(
    artist: ArtistOrSeq, use_pyproj: bool = True, strict: bool = False
) -> None:
    '''
    用中国国界裁剪 Artist

    Parameters
    ----------
    artist : ArtistOrSeq
        被裁剪的Artist对象。可以返回自以下方法：
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    strict : bool, optional
        是否使用更严格的裁剪方法。默认为 False。
        为 True 时即便 GeoAxes 的边界不是矩形也能避免出界。
    '''
    clip_by_polygon(
        artist=artist,
        polygon=fshp.get_cn_border(),
        use_pyproj=use_pyproj,
        strict=strict,
    )


def clip_by_cn_province(
    artist: ArtistOrSeq,
    province: str,
    use_pyproj: bool = True,
    strict: bool = False,
) -> None:
    '''
    用中国省界裁剪 Artist

    Parameters
    ----------
    artist : ArtistOrSeq
        被裁剪的Artist对象。可以返回自以下方法：
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    province : str
        单个省名

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    strict : bool, optional
        是否使用更严格的裁剪方法。默认为 False。
        为 True 时即便 GeoAxes 的边界不是矩形也能避免出界。
    '''
    if not isinstance(province, str):
        raise ValueError('只支持单个省')
    clip_by_polygon(
        artist=artist,
        polygon=fshp.get_cn_province(province),
        use_pyproj=use_pyproj,
        strict=strict,
    )


def clip_by_cn_city(
    artist: ArtistOrSeq,
    city: str = None,
    use_pyproj: bool = True,
    strict: bool = False,
) -> None:
    '''
    用中国市界裁剪 Artist

    Parameters
    ----------
    artist : ArtistOrSeq
        被裁剪的Artist对象。可以返回自以下方法：
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    city : str
        单个市名

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    strict : bool, optional
        是否使用更严格的裁剪方法。默认为 False。
        为 True 时即便 GeoAxes 的边界不是矩形也能避免出界。
    '''
    if is_sequence(city):
        raise ValueError('只支持单个市')
    clip_by_polygon(
        artist=artist,
        polygon=fshp.get_cn_city(city),
        use_pyproj=use_pyproj,
        strict=strict,
    )


def clip_by_land(
    artist: ArtistOrSeq, use_pyproj: bool = True, strict: bool = False
) -> None:
    '''
    用陆地边界裁剪 Artist

    注意全球数据可能在地图边界出现错误的结果。

    Parameters
    ----------
    artist : ArtistOrSeq
        被裁剪的Artist对象。可以返回自以下方法：
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    strict : bool, optional
        是否使用更严格的裁剪方法。默认为 False。
        为 True 时即便 GeoAxes 的边界不是矩形也能避免出界。
    '''
    clip_by_polygon(
        artist=artist,
        polygon=fshp.get_land(),
        use_pyproj=use_pyproj,
        strict=strict,
    )


def clip_by_ocean(
    artist: ArtistOrSeq, use_pyproj: bool = True, strict: bool = False
) -> None:
    '''
    用海洋边界裁剪 Artist

    注意全球数据可能在地图边界出现错误的结果。

    Parameters
    ----------
    artist : ArtistOrSeq
        被裁剪的Artist对象。可以返回自以下方法：
        - plot, scatter
        - contour, contourf, clabel
        - pcolor, pcolormesh
        - imshow
        - quiver

    use_pyproj : bool, optional
        是否直接用 pyproj 做坐标变换。默认为 True，速度更快但也效果也容易出错。

    strict : bool, optional
        是否使用更严格的裁剪方法。默认为 False。
        为 True 时即便 GeoAxes 的边界不是矩形也能避免出界。
    '''
    clip_by_polygon(
        artist=artist,
        polygon=fshp.get_ocean(),
        use_pyproj=use_pyproj,
        strict=strict,
    )


def add_texts(
    ax: Axes, x: Any, y: Any, s: Sequence[str], **kwargs: Any
) -> list[Text]:
    '''
    在 Axes 上添加一组文本

    Parameters
    ----------
    ax : Axes
        目标Axes

    x : (n,) array_like
        文本的横坐标数组

    y : (n,) array_like
        文本的纵坐标数组

    s : (n,) sequence of str
        一组字符串文本

    **kwargs
        Axes.text 方法的关键字参数
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

    Returns
    -------
    texts : (n,) list of Text
        一组 Text 对象
    '''
    x = x if is_sequence(x) else [x]
    y = y if is_sequence(y) else [y]
    s = s if is_sequence(s) else [s]
    if not len(x) == len(y) == len(s):
        raise ValueError('x、y 和 s 长度必须相同')

    kwargs = normalize_kwargs(kwargs, Text)
    kwargs.setdefault('horizontalalignment', 'center')
    kwargs.setdefault('verticalalignment', 'center')
    kwargs.setdefault('clip_on', True)
    texts = [ax.text(xi, yi, si, **kwargs) for xi, yi, si in zip(x, y, s)]

    return texts


def _add_cn_texts(
    ax: Axes,
    lons: Any,
    lats: Any,
    names: Sequence[str],
    **kwargs: Any,
) -> list[Text]:
    '''在 Axes 上标注中国地名'''
    kwargs = normalize_kwargs(kwargs, Text)
    kwargs.setdefault('fontsize', 'x-small')
    if isinstance(ax, GeoAxes):
        kwargs.setdefault('clip_box', ax.bbox)
        kwargs.setdefault('transform', PlateCarree())

    # 默认使用精简的思源黑体
    filepath = DATA_DIRPATH / 'zh_font.otf'
    if kwargs.get('fontname') is None and kwargs.get('fontfamily') is None:
        kwargs.setdefault('fontproperties', filepath)

    return add_texts(ax, lons, lats, names, **kwargs)


def label_cn_province(
    ax: Axes,
    province: Optional[StrOrSeq] = None,
    short: bool = True,
    **kwargs: Any,
) -> list[Text]:
    '''
    在 Axes 上标注中国省名

    Parameters
    ----------
    ax : Axes
        目标 Axes

    province : StrOrSeq, optional
        单个或一组省名。默认为 None，表示获取所有省。

    short : bool, optional
        是否使用缩短的省名。默认为 True。

    **kwargs
        Axes.text 方法的关键字参数
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

    Returns
    -------
    texts : list of Text
        表示省名的 Text 对象
    '''
    locs = fshp._get_pr_locs(province)
    table = fshp._PR_TABLE.iloc[locs]
    lons = table['lon']
    lats = table['lat']
    names = table['short_name' if short else 'pr_name']
    texts = _add_cn_texts(ax, lons, lats, names, **kwargs)

    return texts


def label_cn_city(
    ax: Axes,
    city: Optional[StrOrSeq] = None,
    province: Optional[StrOrSeq] = None,
    short: bool = True,
    **kwargs: Any,
) -> list[Text]:
    '''
    在 Axes 上标注中国市名

    Parameters
    ----------
    ax : Axes
        目标 Axes

    city : StrOrSeq, optional
        单个或一组市名。默认为 None，表示获取所有市。

    province : StrOrSeq, optional
        单个或一组省名，获取属于某个省的所有市。
        默认为 None，表示不指定省名。
        不能同时指定 city 和 province。

    short : bool, optional
        是否使用缩短的市名。默认为 True。

    **kwargs
        Axes.text 方法的关键字参数
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

    Returns
    -------
    texts : list of Text
        表示市名的 Text 对象
    '''
    locs = fshp._get_ct_locs(city, province)
    table = fshp._CT_TABLE.iloc[locs]
    lons = table['lon']
    lats = table['lat']
    names = table['short_name' if short else 'ct_name']
    texts = _add_cn_texts(ax, lons, lats, names, **kwargs)

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
    crs = PlateCarree()
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
    crs = PlateCarree()
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
            lon_line = sgeom.LineString(np.c_[lons, lats])
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
            lat_line = sgeom.LineString(np.c_[lons, lats])
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
    *,
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
        if isinstance(ax.projection, (PlateCarree, Mercator)):
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


def quick_cn_map(
    extents: Optional[Any] = None,
    use_geoaxes: bool = True,
    figsize: Optional[tuple[float, float]] = None,
) -> Axes:
    '''
    快速制作带省界和九段线的中国地图.

    Parameters
    ----------
    extents : extents : (4,) array_like, optional
        经纬度范围[lon0, lon1, lat0, lat1]. 默认为None, 表示[70, 140, 0, 60].

    use_geoaxes : bool, optional
        是否使用GeoAxes. 默认为True.

    figsize : (2,) tuple of int, optional
        Figure宽高. 默认为None, 表示(6.4, 4.8).

    Returns
    -------
    ax : Axes
        表示地图的Axes
    '''
    if extents is None:
        extents = [70, 140, 10, 60]
    fig = plt.figure(figsize=figsize)
    if use_geoaxes:
        crs = PlateCarree()
        ax = fig.add_subplot(projection=crs)
    else:
        ax = fig.add_subplot()
        ax.set_aspect(1)
    set_map_ticks(ax, extents)
    add_cn_province(ax)
    add_nine_line(ax)

    return ax


def add_quiver_legend(
    Q: Quiver,
    U: float,
    units: str = 'm/s',
    width: float = 0.15,
    height: float = 0.15,
    loc: Literal[
        'lower left', 'lower right', 'upper left', 'upper right'
    ] = 'lower right',
    qk_kwargs: Optional[dict] = None,
    patch_kwargs: Optional[dict] = None,
) -> QuiverLegend:
    '''
    在Axes上添加Quiver的图例(带矩形背景的QuiverKey).

    箭头下方有形如'{U} {units}'的标签.

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

    loc : {'lower left', 'lower right', 'upper left', 'upper right'}, optional
        图例位置. 默认为'lower right'.

    qk_kwargs : dict, optional
        QuiverKey类的关键字参数. 默认为None.
        例如labelsep, labelcolor, fontproperties等.
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiverkey.html

    patch_kwargs : dict, optional
        表示背景方框的Rectangle类的关键字参数. 默认为None.
        例如linewidth, edgecolor, facecolor等.
        https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html

    Returns
    -------
    quiver_legend : QuiverLegend
        图例对象.
    '''
    quiver_legend = QuiverLegend(
        Q, U, units, width, height, loc, qk_kwargs, patch_kwargs
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
    pc_kwargs: Optional[dict] = None,
    text_kwargs: Optional[dict] = None,
) -> Compass:
    '''
    在Axes上添加指北针.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    x, y : float
        指北针的横纵坐标. 基于Axes坐标系.

    angle : float, optional
        指北针的方位角. 单位为度.
        默认为None, 表示GeoAxes会自动计算角度, 而Axes默认正北.

    size : float, optional
        指北针大小. 单位为点, 默认为20.

    style : {'arrow', 'circle', 'star'}, optional
        指北针造型. 默认为'arrow'.

    pc_kwargs : dict, optional
        表示指北针的PathCollection类的关键字参数. 默认为None.
        例如linewidth, edgecolor, facecolor等.
        https://matplotlib.org/stable/api/collections_api.html

    text_kwargs : dict, optional
        表示指北针N字的Text类的关键字参数. 默认为None.
        https://matplotlib.org/stable/api/text_api.html

    Returns
    -------
    compass : Compass
        指北针对象.
    '''
    compass = Compass(x, y, angle, size, style, pc_kwargs, text_kwargs)
    ax.add_artist(compass)

    return compass


def add_scale_bar(
    ax: Axes,
    x: float,
    y: float,
    length: float = 1000,
    units: Literal['m', 'km'] = 'km',
) -> ScaleBar:
    '''
    在Axes上添加地图比例尺.

    会根据Axes的投影计算比例尺大小.
    假设Axes的投影是PlateCarree.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    x, y : float
        比例尺左端的横纵坐标. 基于Axes坐标系.

    length : float, optional
        比例尺长度. 默认为1000.

    units : {'m', 'km'}, optional
        比例尺长度的单位. 默认为'km'.

    Returns
    -------
    scale_bar : ScaleBar
        比例尺对象. 刻度可以通过set_xticks方法修改.
    '''
    return ScaleBar(ax, x, y, length, units)


def add_frame(ax: Axes, width: float = 5, **kwargs: Any) -> Frame:
    '''
    在Axes上添加GMT风格的边框.

    Parameters
    ----------
    ax : Axes
        目标Axes. 目前仅支持PlateCarree和Mercator投影的GeoAxes.

    width : float, optional
        边框的宽度. 单位为点, 默认为5.

    **kwargs:
        表示边框的PathCollection类的关键字参数.
        例如linewidth, edgecolor, facecolor等.
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    frame : Frame
        边框对象.
    '''
    frame = Frame(width, **kwargs)
    ax.add_artist(frame)

    return frame


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
    path = rectangle_path(*extents).interpolated(steps)
    patch = PathPatch(path, **kwargs)
    ax.add_patch(patch)

    return patch


# TODO: ax在new_ax之前绘制?
def add_mini_axes(
    ax: Axes,
    shrink: float = 0.4,
    aspect: float = 1,
    loc: Literal[
        'lower left', 'lower right', 'upper left', 'upper right'
    ] = 'lower right',
    projection: Optional[CRS] = None,
) -> Axes:
    '''
    在Axes的角落添加一个迷你Axes并返回.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    shrink : float, optional
        缩小倍数. 默认为0.4.

    aspect : float, optional
        单位坐标的高宽比. 默认为 1, 与GeoAxes相同.

    loc : {'lower left', 'lower right', 'upper left', 'upper right'}, optional
        指定放置在哪个角落. 默认为'lower right'.

    projection : CRS, optional
        新Axes的投影. 默认为None, 表示沿用ax的投影.

    Returns
    -------
    new_ax : Axes
        新的迷你Axes.
    '''
    if projection is None:
        if isinstance(ax, GeoAxes):
            projection = ax.projection
    new_ax = ax.figure.add_subplot(projection=projection)
    new_ax.set_aspect(aspect)
    draw = new_ax.draw

    def new_draw(renderer: RendererBase) -> None:
        '''在原先的draw前调整new_ax的大小位置.'''
        bbox = ax.get_position()
        new_bbox = new_ax.get_position()
        # shrink=1时new_ax恰好有一边填满ax.
        if bbox.width / bbox.height < new_bbox.width / new_bbox.height:
            ratio = bbox.width / new_bbox.width * shrink
        else:
            ratio = bbox.height / new_bbox.height * shrink
        width = new_bbox.width * ratio
        height = new_bbox.height * ratio

        if loc == 'lower left':
            x0 = bbox.x0
            x1 = bbox.x0 + width
            y0 = bbox.y0
            y1 = bbox.y0 + height
        elif loc == 'lower right':
            x0 = bbox.x1 - width
            x1 = bbox.x1
            y0 = bbox.y0
            y1 = bbox.y0 + height
        elif loc == 'upper left':
            x0 = bbox.x0
            x1 = bbox.x0 + width
            y0 = bbox.y1 - height
            y1 = bbox.y1
        elif loc == 'upper right':
            x0 = bbox.x1 - width
            x1 = bbox.x1
            y0 = bbox.y1 - height
            y1 = bbox.y1
        else:
            raise ValueError(
                "loc只能取{'lower left', 'lower right', 'upper left', 'upper right'}"
            )

        new_bbox = Bbox.from_extents(x0, y0, x1, y1)
        new_ax.set_position(new_bbox)
        draw(renderer)

    new_ax.draw = new_draw

    return new_ax


# TODO: mpl_toolkits.axes_grid1实现.
def add_side_axes(
    ax: Any,
    loc: Literal['left', 'right', 'bottom', 'top'],
    pad: float,
    width: float,
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

    width : float
        新Axes的宽度(高度). 基于Figure坐标系.

    Returns
    -------
    new_ax : Axes
        新Axes对象.
    '''
    # 获取一组Axes的位置.
    axs = np.atleast_1d(ax).ravel()
    bbox = Bbox.union([ax.get_position() for ax in axs])

    # 可选四个方向.
    if loc == 'left':
        x0 = bbox.x0 - pad - width
        x1 = x0 + width
        y0 = bbox.y0
        y1 = bbox.y1
    elif loc == 'right':
        x0 = bbox.x1 + pad
        x1 = x0 + width
        y0 = bbox.y0
        y1 = bbox.y1
    elif loc == 'bottom':
        x0 = bbox.x0
        x1 = bbox.x1
        y0 = bbox.y0 - pad - width
        y1 = y0 + width
    elif loc == 'top':
        x0 = bbox.x0
        x1 = bbox.x1
        y0 = bbox.y1 + pad
        y1 = y0 + width
    else:
        raise ValueError("loc只能取{'left', 'right', 'bottom', 'top'}")
    new_bbox = Bbox.from_extents(x0, y0, x1, y1)
    new_ax = axs[0].figure.add_axes(new_bbox)

    return new_ax


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


def load_test_data() -> NpzFile:
    '''读取测试用的数据. 包含地表2m气温(K)和水平10m风速.'''
    filepath = DATA_DIRPATH / 'test.npz'
    return np.load(str(filepath))


def savefig(fname: Any, fig: Optional[Figure] = None, **kwargs) -> None:
    '''保存Figure为图片.'''
    if fig is None:
        fig = plt.gcf()
    kwargs.setdefault('dpi', 300)
    kwargs.setdefault('bbox_inches', 'tight')
    fig.savefig(fname, **kwargs)


@deprecator(add_scale_bar)
def add_map_scale(*args, **kwargs):
    return add_scale_bar(*args, **kwargs)


@deprecator(add_frame)
def gmt_style_frame(*args, **kwargs):
    return add_frame(*args, **kwargs)


@deprecator(add_mini_axes, raise_error=True)
def move_axes_to_corner():
    pass
