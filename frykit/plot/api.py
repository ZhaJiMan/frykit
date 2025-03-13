from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from functools import wraps
from typing import Any, Literal, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shapely
from cartopy.crs import CRS, Mercator, PlateCarree
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from frykit_data import DATA_DIRPATH
from matplotlib import font_manager
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.cbook import normalize_kwargs
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.quiver import Quiver
from matplotlib.text import Text
from matplotlib.ticker import Formatter
from matplotlib.transforms import Bbox
from numpy.lib.npyio import NpzFile
from numpy.typing import ArrayLike, NDArray
from shapely.geometry.base import BaseGeometry

from frykit.calc import asarrays, lon_to_180
from frykit.option import DataSource, get_option, validate_option
from frykit.plot.artist import (
    Compass,
    Frame,
    GeometryPathCollection,
    QuiverLegend,
    ScaleBar,
    TextCollection,
    _get_axes_extents,
    _get_ticks_between,
    _resolve_fast_transform,
)
from frykit.plot.projection import PLATE_CARREE
from frykit.plot.utils import box_path
from frykit.shp.data import (
    LineName,
    NameOrAdcode,
    _get_cn_city_locs,
    _get_cn_city_table,
    _get_cn_district_locs,
    _get_cn_district_table,
    _get_cn_province_locs,
    _get_cn_province_table,
    get_cn_border,
    get_cn_city,
    get_cn_district,
    get_cn_line,
    get_cn_province,
    get_countries,
    get_land,
    get_ocean,
)
from frykit.shp.typing import PolygonType
from frykit.utils import deprecator, format_literal_error, format_type_error, to_list

__all__ = [
    "add_geometries",
    "add_cn_province",
    "add_cn_city",
    "add_cn_district",
    "add_cn_border",
    "add_cn_line",
    "add_countries",
    "add_land",
    "add_ocean",
    "add_texts",
    "label_cn_province",
    "label_cn_city",
    "label_cn_district",
    "clip_by_polygon",
    "clip_by_cn_province",
    "clip_by_cn_city",
    "clip_by_cn_district",
    "clip_by_cn_border",
    "clip_by_land",
    "clip_by_ocean",
    "set_map_ticks",
    "quick_cn_map",
    "add_quiver_legend",
    "add_compass",
    "add_scale_bar",
    "add_frame",
    "add_box",
    "add_mini_axes",
    "get_cross_section_xticks",
    "make_qualitative_palette",
    "get_aod_cmap",
    "CenteredBoundaryNorm",
    "plot_colormap",
    "letter_axes",
    "load_test_data",
    "savefig",
    "get_font_names",
    "add_geoms",
    "get_qualitative_palette",
]


# TODO: 用 scatter 绘制 Point
def add_geometries(
    ax: Axes,
    geometries: BaseGeometry | Sequence[BaseGeometry],
    crs: CRS | None = None,
    fast_transform: bool | None = None,
    skip_outside: bool | None = None,
    **kwargs: Any,
) -> GeometryPathCollection:
    """
    将几何对象添加到 Axes 上

    BUG: Point 和 MultiPoint 画不出来

    Parameters
    ----------
    ax : Axes
        目标 Axes

    geometries : BaseGeometry or sequence of BaseGeometry
        一个或一组几何对象

    crs : CRS, optional
        当 ax 是 Axes 时 crs 只能为 None，表示不做变换。
        当 ax 是 GeoAxes 时会将 geometries 从 crs 坐标系变换到 ax.projection 坐标系上。
        此时默认值 None 表示 PlateCarree 投影。

    fast_transform : bool, default None
        是否直接用 pyproj 做坐标变换，速度更快但也更容易出错。
        默认为 None，表示使用默认的全局配置（True）。

    skip_outside : bool, default None
        是否跳过 ax 边框外的几何对象，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    GeometryPathCollection
        表示几何对象的集合对象

    See Also
    --------
    cartopy.mpl.geoaxes.GeoAxes.add_geometries
    """
    geometries = cast(list[BaseGeometry], to_list(geometries))
    return GeometryPathCollection(
        ax=ax,
        geometries=geometries,
        crs=crs,
        fast_transform=fast_transform,
        skip_outside=skip_outside,
        **kwargs,
    )


def _set_pc_kwargs(kwargs: dict) -> dict:
    """设置 add_cn_xxx 系列函数的 kwargs"""
    kwargs = normalize_kwargs(kwargs, PathCollection)
    kwargs.setdefault("linewidth", 0.5)
    kwargs.setdefault("edgecolor", "k")
    kwargs.setdefault("facecolor", "none")
    kwargs.setdefault("zorder", 1.5)

    return kwargs


def add_cn_province(
    ax: Axes,
    province: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    fast_transform: bool | None = None,
    skip_outside: bool | None = None,
    data_source: DataSource | None = None,
    **kwargs: Any,
) -> GeometryPathCollection:
    """
    在 Axes 上添加中国省界

    Parameters
    ----------
    ax : Axes
        目标 Axes

    province : NameOrAdcode or sequence of NameOrAdcode, default None
        省名或 adcode。可以是多个省。默认为 None，表示所有省。

    fast_transform : bool, default None
        是否直接用 pyproj 做坐标变换，速度更快但也更容易出错。
        默认为 None，表示使用默认的全局配置（True）。

    skip_outside : bool, default None
        是否跳过 ax 边框外的几何对象，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    data_source : {'amap', 'tianditu'}, default None
        数据源。默认为 None，表示使用默认的全局配置（amap）。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    GeometryPathCollection
        表示几何对象的集合对象
    """
    return add_geometries(
        ax=ax,
        geometries=get_cn_province(province, data_source),
        fast_transform=fast_transform,
        skip_outside=skip_outside,
        **_set_pc_kwargs(kwargs),
    )


def add_cn_city(
    ax: Axes,
    city: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    province: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    fast_transform: bool | None = None,
    skip_outside: bool | None = None,
    data_source: DataSource | None = None,
    **kwargs: Any,
) -> GeometryPathCollection:
    """
    在 Axes 上添加中国市界

    Parameters
    ----------
    ax : Axes
        目标 Axes

    city : NameOrAdcode or sequence of NameOrAdcode, default None
        市名或 adcode。可以是多个市。默认为 None，表示所有市。

    province : NameOrAdcode or sequence of NameOrAdcode, default None
        省名或 adcode。表示指定某个省的所有市。可以是多个省。
        默认为 None，表示不指定省。

    fast_transform : bool, default None
        是否直接用 pyproj 做坐标变换，速度更快但也更容易出错。
        默认为 None，表示使用默认的全局配置（True）。

    skip_outside : bool, default None
        是否跳过 ax 边框外的几何对象，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    data_source : {'amap', 'tianditu'}, default None
        数据源。默认为 None，表示使用默认的全局配置（amap）。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    GeometryPathCollection
        表示几何对象的集合对象
    """
    return add_geometries(
        ax=ax,
        geometries=get_cn_city(city, province, data_source),
        fast_transform=fast_transform,
        skip_outside=skip_outside,
        **_set_pc_kwargs(kwargs),
    )


def add_cn_district(
    ax: Axes,
    district: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    city: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    province: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    fast_transform: bool | None = None,
    skip_outside: bool | None = None,
    data_source: DataSource | None = None,
    **kwargs: Any,
) -> GeometryPathCollection:
    """
    在 Axes 上添加中国县界

    Parameters
    ----------
    ax : Axes
        目标 Axes

    district : NameOrAdcode or sequence of NameOrAdcode, default None
        县名或 adcode。可以是多个县。默认为 None，表示所有县。

    city : NameOrAdcode or sequence of NameOrAdcode, default None
        市名或 adcode。表示指定某个市的所有县。可以是多个市。
        默认为 None，表示不指定市。

    province : NameOrAdcode or sequence of NameOrAdcode, default None
        省名或 adcode。表示指定某个省的所有县。可以是多个省。
        默认为 None，表示不指定省。

    fast_transform : bool, default None
        是否直接用 pyproj 做坐标变换，速度更快但也更容易出错。
        默认为 None，表示使用默认的全局配置（True）。

    skip_outside : bool, default None
        是否跳过 ax 边框外的几何对象，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    data_source : {'amap', 'tianditu'}, default None
        数据源。默认为 None，表示使用默认的全局配置（amap）。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    GeometryPathCollection
        表示几何对象的集合对象
    """
    return add_geometries(
        ax=ax,
        geometries=get_cn_district(district, city, province, data_source),
        fast_transform=fast_transform,
        skip_outside=skip_outside,
        **_set_pc_kwargs(kwargs),
    )


def add_cn_border(
    ax: Axes,
    fast_transform: bool | None = None,
    skip_outside: bool | None = None,
    data_source: DataSource | None = None,
    **kwargs: Any,
) -> GeometryPathCollection:
    """
    在 Axes 上添加中国国界

    Parameters
    ----------
    ax : Axes
        目标 Axes

    fast_transform : bool, default None
        是否直接用 pyproj 做坐标变换，速度更快但也更容易出错。
        默认为 None，表示使用默认的全局配置（True）。

    skip_outside : bool, default None
        是否跳过 ax 边框外的几何对象，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    data_source : {'amap', 'tianditu'}, default None
        数据源。默认为 None，表示使用默认的全局配置（amap）。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    GeometryPathCollection
        表示几何对象的集合对象
    """
    return add_geometries(
        ax=ax,
        geometries=get_cn_border(data_source),
        fast_transform=fast_transform,
        skip_outside=skip_outside,
        **_set_pc_kwargs(kwargs),
    )


def add_cn_line(
    ax: Axes,
    name: LineName | Sequence[LineName] = "九段线",
    fast_transform: bool | None = None,
    skip_outside: bool | None = None,
    **kwargs: Any,
) -> GeometryPathCollection:
    """
    在 Axes 上添加中国的修饰线段

    Parameters
    ----------
    ax : Axes
        目标 Axes

    name : {'省界', '特别行政区界', '九段线', '未定国界'} or sequence of str, default '九段线'
        线段名称。可以是多种线段。默认为 '九段线'。

    fast_transform : bool, default None
        是否直接用 pyproj 做坐标变换，速度更快但也更容易出错。
        默认为 None，表示使用默认的全局配置（True）。

    skip_outside : bool, default None
        是否跳过 ax 边框外的几何对象，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    GeometryPathCollection
        表示几何对象的集合对象
    """
    return add_geometries(
        ax=ax,
        geometries=get_cn_line(name),
        fast_transform=fast_transform,
        skip_outside=skip_outside,
        **_set_pc_kwargs(kwargs),
    )


def add_countries(
    ax: Axes,
    fast_transform: bool | None = None,
    skip_outside: bool | None = None,
    **kwargs: Any,
) -> GeometryPathCollection:
    """
    在 Axes 上添加所有国家的国界

    Parameters
    ----------
    ax : Axes
        目标 Axes

    fast_transform : bool, default None
        是否直接用 pyproj 做坐标变换，速度更快但也更容易出错。
        默认为 None，表示使用默认的全局配置（True）。

    skip_outside : bool, default None
        是否跳过 ax 边框外的几何对象，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    GeometryPathCollection
        表示几何对象的集合对象
    """
    return add_geometries(
        ax=ax,
        geometries=get_countries(),
        fast_transform=fast_transform,
        skip_outside=skip_outside,
        **_set_pc_kwargs(kwargs),
    )


def add_land(
    ax: Axes,
    fast_transform: bool | None = None,
    skip_outside: bool | None = None,
    **kwargs: Any,
) -> GeometryPathCollection:
    """
    在 Axes 上添加陆地

    注意全球数据可能在地图边界出现错误的结果

    Parameters
    ----------
    ax : Axes
        目标 Axes

    fast_transform : bool, default None
        是否直接用 pyproj 做坐标变换，速度更快但也更容易出错。
        默认为 None，表示使用默认的全局配置（True）。

    skip_outside : bool, default None
        是否跳过 ax 边框外的几何对象，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    GeometryPathCollection
        表示几何对象的集合对象
    """
    return add_geometries(
        ax=ax,
        geometries=get_land(),
        fast_transform=fast_transform,
        skip_outside=skip_outside,
        **_set_pc_kwargs(kwargs),
    )


def add_ocean(
    ax: Axes,
    fast_transform: bool | None = None,
    skip_outside: bool | None = None,
    **kwargs: Any,
) -> GeometryPathCollection:
    """
    在 Axes 上添加海洋

    注意全球数据可能在地图边界出现错误的结果

    Parameters
    ----------
    ax : Axes
        目标 Axes

    fast_transform : bool, default None
        是否直接用 pyproj 做坐标变换，速度更快但也更容易出错。
        默认为 None，表示使用默认的全局配置（True）。

    skip_outside : bool, default None
        是否跳过 ax 边框外的几何对象，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    **kwargs
        PathCollection 类的关键字参数。
        例如 edgecolor、facecolor、cmap、norm 和 array 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    GeometryPathCollection
        表示几何对象的集合对象
    """
    return add_geometries(
        ax=ax,
        geometries=get_ocean(),
        fast_transform=fast_transform,
        skip_outside=skip_outside,
        **_set_pc_kwargs(kwargs),
    )


def add_texts(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    s: ArrayLike,
    skip_outside: bool | None = None,
    **kwargs: Any,
) -> TextCollection:
    """
    在 Axes 上添加一组文本

    Parameters
    ----------
    ax : Axes
        目标Axes

    x, y : (n,) array_like of float
        文本的横纵坐标

    s : (n,) array_like of str
        文本字符串

    skip_outside : bool, default None
        是否跳过 ax 边框外的几何对象，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    **kwargs
        Text 类的关键字参数。
        例如 fontsize、fontfamily 和 color 等。
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

    Returns
    -------
    TextCollection
        表示 Text 的集合对象
    """
    if not isinstance(ax, Axes):
        raise TypeError(format_type_error("ax", ax, Axes))

    tc = TextCollection(x, y, s, skip_outside, **kwargs)
    ax.add_artist(tc)
    for text in tc.texts:
        text.axes = ax
        if not text.is_transform_set():
            text.set_transform(ax.transData)
        text.set_clip_box(ax.bbox)

    return tc


def _add_cn_texts(
    ax: Axes,
    lons: ArrayLike,
    lats: ArrayLike,
    names: ArrayLike,
    skip_outside: bool | None = None,
    **kwargs: Any,
) -> TextCollection:
    """用 add_texts 函数添加中国地名"""
    kwargs = normalize_kwargs(kwargs, Text)
    kwargs.setdefault("fontsize", "small")
    if isinstance(ax, GeoAxes):
        kwargs.setdefault("transform", PLATE_CARREE)

    # 用户不修改字体时使用自带的中文字体
    if (
        kwargs.get("fontname") is None
        and kwargs.get("fontfamily") is None
        and mpl.rcParams["font.family"][0] == "sans-serif"
        and mpl.rcParams["font.sans-serif"][0] == "DejaVu Sans"
    ):
        filepath = DATA_DIRPATH / "zh_font.otf"
        kwargs.setdefault("fontproperties", filepath)

    return add_texts(
        ax=ax,
        x=lons,
        y=lats,
        s=names,
        skip_outside=skip_outside,
        **kwargs,
    )


def label_cn_province(
    ax: Axes,
    province: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    short_name: bool = True,
    skip_outside: bool | None = None,
    data_source: DataSource | None = None,
    **kwargs: Any,
) -> TextCollection:
    """
    在 Axes 上标注中国省名

    Parameters
    ----------
    ax : Axes
        目标 Axes

    province : NameOrAdcode or sequence of NameOrAdcode, default None
        省名或 adcode。可以是多个省。默认为 None，表示所有省。

    short_name : bool, default True
        是否使用缩短的名称。默认为 True。

    skip_outside : bool, default None
        是否跳过 ax 边框外的文本，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    **kwargs
        Text 类的关键字参数。
        例如 fontsize、fontfamily 和 color 等。
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

    Returns
    -------
    TextCollection
        表示 Text 的集合对象
    """
    df = _get_cn_province_table(data_source)
    locs = _get_cn_province_locs(province, data_source)
    key = "short_name" if short_name else "province_name"
    df = df.iloc[locs]

    return _add_cn_texts(
        ax=ax,
        lons=df["lon"],
        lats=df["lat"],
        names=df[key],
        skip_outside=skip_outside,
        **kwargs,
    )


def label_cn_city(
    ax: Axes,
    city: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    province: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    short_name: bool = True,
    skip_outside: bool | None = None,
    data_source: DataSource | None = None,
    **kwargs: Any,
) -> TextCollection:
    """
    在 Axes 上标注中国市名

    Parameters
    ----------
    ax : Axes
        目标 Axes

    city : NameOrAdcode or sequence of NameOrAdcode, default None
        市名或 adcode。可以是多个市。默认为 None，表示所有市。

    province : NameOrAdcode or sequence of NameOrAdcode, default None
        省名或 adcode。表示指定某个省的所有市。可以是多个省。
        默认为 None，表示不指定省。

    short_name : bool, default True
        是否使用缩短的名称。默认为 True。

    skip_outside : bool, default None
        是否跳过 ax 边框外的文本，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    **kwargs
        Text 类的关键字参数。
        例如 fontsize、fontfamily 和 color 等。
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

    Returns
    -------
    TextCollection
        表示 Text 的集合对象
    """
    df = _get_cn_city_table(data_source)
    locs = _get_cn_city_locs(city, province, data_source)
    key = "short_name" if short_name else "city_name"
    df = df.iloc[locs]

    return _add_cn_texts(
        ax=ax,
        lons=df["lon"],
        lats=df["lat"],
        names=df[key],
        skip_outside=skip_outside,
        **kwargs,
    )


def label_cn_district(
    ax: Axes,
    district: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    city: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    province: NameOrAdcode | Sequence[NameOrAdcode] | None = None,
    short_name: bool = True,
    skip_outside: bool | None = None,
    data_source: DataSource | None = None,
    **kwargs: Any,
) -> TextCollection:
    """
    在 Axes 上标注中国县名

    Parameters
    ----------
    ax : Axes
        目标 Axes

    district : NameOrAdcode or sequence of NameOrAdcode, default None
        县名或 adcode。可以是多个县。默认为 None，表示所有县。

    city : NameOrAdcode or sequence of NameOrAdcode, default None
        市名或 adcode。表示指定某个市的所有县。可以是多个市。
        默认为 None，表示不指定市。

    province : NameOrAdcode or sequence of NameOrAdcode, default None
        省名或 adcode。表示指定某个省的所有县。可以是多个省。
        默认为 None，表示不指定省。

    short_name : bool, default True
        是否使用缩短的名称。默认为 True。

    skip_outside : bool, default None
        是否跳过 ax 边框外的文本，提高局部绘制速度。
        默认为 None，表示使用默认的全局配置（True）。

    **kwargs
        Text 类的关键字参数。
        例如 fontsize、fontfamily 和 color 等。
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

    Returns
    -------
    TextCollection
        表示 Text 的集合对象
    """
    df = _get_cn_district_table(data_source)
    locs = _get_cn_district_locs(district, city, province, data_source)
    key = "short_name" if short_name else "district_name"
    df = df.iloc[locs]

    return _add_cn_texts(
        ax=ax,
        lons=df["lon"],
        lats=df["lat"],
        names=df[key],
        skip_outside=skip_outside,
        **kwargs,
    )


def _resolve_strict_clip(strict_clip: bool | None) -> bool:
    if strict_clip is None:
        return get_option("strict_clip")
    else:
        validate_option("strict_clip", strict_clip)
        return strict_clip


def clip_by_polygon(
    artist: Artist | Iterable[Artist],
    polygon: PolygonType | Iterable[PolygonType],
    crs: CRS | None = None,
    ax: Axes | None = None,
    fast_transform: bool | None = None,
    strict_clip: bool | None = None,
) -> None:
    fast_transform = _resolve_fast_transform(fast_transform)
    strict_clip = _resolve_strict_clip(strict_clip)


def clip_by_cn_province(): ...


def clip_by_cn_city(): ...


def clip_by_cn_district(): ...


def clip_by_cn_border(): ...


def clip_by_land(): ...


def clip_by_ocean(): ...


def _set_axes_ticks(
    ax: Axes,
    extents: tuple[float, float, float, float] | None,
    major_xticks: NDArray,
    major_yticks: NDArray,
    minor_xticks: NDArray,
    minor_yticks: NDArray,
    xformatter: Formatter,
    yformatter: Formatter,
) -> None:
    """设置 PlateCarree 投影的普通 Axes 的刻度"""
    if extents is None:
        extents = (-180, 180, -90, 90)
    lon0, lon1, lat0, lat1 = extents
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)

    # 避免不可见的刻度影响速度
    major_xticks = _get_ticks_between(major_xticks, lon0, lon1)
    major_yticks = _get_ticks_between(major_yticks, lat0, lat1)
    minor_xticks = _get_ticks_between(minor_xticks, lon0, lon1)
    minor_yticks = _get_ticks_between(minor_yticks, lat0, lat1)

    ax.set_xticks(major_xticks, minor=False)
    ax.set_yticks(major_yticks, minor=False)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)


def _set_simple_geoaxes_ticks(
    ax: GeoAxes,
    extents: tuple[float, float, float, float] | None,
    major_xticks: NDArray,
    major_yticks: NDArray,
    minor_xticks: NDArray,
    minor_yticks: NDArray,
    xformatter: Formatter,
    yformatter: Formatter,
) -> None:
    """设置 PlateCarree 和 Mercator 投影的 GeoAxes 的刻度"""
    if extents is not None:
        ax.set_extent(extents, crs=PLATE_CARREE)
    else:
        ax.set_global()
        lat0, lat1 = -90, 90
        if isinstance(ax.projection, Mercator):
            lat0, lat1 = PLATE_CARREE.transform_points(
                ax.projection,
                np.array(ax.projection.x_limits),
                np.array(ax.projection.y_limits),
            )[:, 1]
        extents = (-180, 180, lat0, lat1)

    lon0, lon1, lat0, lat1 = extents
    major_xticks = _get_ticks_between(major_xticks, lon0, lon1)
    major_yticks = _get_ticks_between(major_yticks, lat0, lat1)
    minor_xticks = _get_ticks_between(minor_xticks, lon0, lon1)
    minor_yticks = _get_ticks_between(minor_yticks, lat0, lat1)

    ax.set_xticks(major_xticks, minor=False, crs=PLATE_CARREE)
    ax.set_yticks(major_yticks, minor=False, crs=PLATE_CARREE)
    ax.set_xticks(minor_xticks, minor=True, crs=PLATE_CARREE)
    ax.set_yticks(minor_yticks, minor=True, crs=PLATE_CARREE)
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)


# TODO: 非矩形边框
def _set_complex_geoaxes_ticks(
    ax: GeoAxes,
    extents: tuple[float, float, float, float] | None,
    major_xticks: NDArray,
    major_yticks: NDArray,
    minor_xticks: NDArray,
    minor_yticks: NDArray,
    xformatter: Formatter,
    yformatter: Formatter,
) -> None:
    """设置非 PlateCarree 和 Mercator 投影的 GeoAxes 的刻度"""
    # 将地图边框设置为矩形
    if extents is None:
        projection_str = str(type(ax.projection)).split("'")[1].split(".")[-1]
        raise ValueError(f"使用 {projection_str} 投影时必须设置 extents 范围")
    ax.set_extent(extents, crs=PLATE_CARREE)

    eps = 1  # 需要稍微扩大一点
    x0, x1, y0, y1 = _get_axes_extents(ax)
    lon0, lon1, lat0, lat1 = ax.get_extent(PLATE_CARREE)
    lon0 = max(-180, lon0 - eps)
    lon1 = min(180, lon1 + eps)
    lat0 = max(-90, lat0 - eps)
    lat1 = min(90, lat1 + eps)

    # 在 data 坐标系用 LineString 表示地图边框的四条边
    lineB = shapely.LineString([(x0, y0), (x1, y0)])
    lineT = shapely.LineString([(x0, y1), (x1, y1)])
    lineL = shapely.LineString([(x0, y0), (x0, y1)])
    lineR = shapely.LineString([(x1, y0), (x1, y1)])

    def get_two_xticks(
        xticks: NDArray, npts: int = 100
    ) -> tuple[list[float], list[float], list[str], list[str]]:
        """获取地图上下边框的 x 刻度和刻度标签"""
        xticksB = []
        xticksT = []
        xticklabelsB = []
        xticklabelsT = []
        lats = np.linspace(lat0, lat1, npts)
        xticks = lon_to_180(xticks)
        xticks = _get_ticks_between(xticks, lon0, lon1)
        for xtick in xticks:
            lons = np.full_like(lats, xtick)
            lon_line = shapely.LineString(np.c_[lons, lats])
            lon_line = ax.projection.project_geometry(lon_line, PLATE_CARREE)
            pointB = lineB.intersection(lon_line)
            if isinstance(pointB, shapely.Point) and not pointB.is_empty:
                xticksB.append(pointB.x)
                xticklabelsB.append(xformatter(xtick))
            pointT = lineT.intersection(lon_line)
            if isinstance(pointT, shapely.Point) and not pointT.is_empty:
                xticksT.append(pointT.x)
                xticklabelsT.append(xformatter(xtick))

        return xticksB, xticksT, xticklabelsB, xticklabelsT

    def get_two_yticks(
        yticks: NDArray, npts: int = 100
    ) -> tuple[list[float], list[float], list[str], list[str]]:
        """获取地图左右边框的 y 刻度和刻度标签"""
        yticksL = []
        yticksR = []
        yticklabelsL = []
        yticklabelsR = []
        lons = np.linspace(lon0, lon1, npts)
        yticks = _get_ticks_between(yticks, lat0, lat1)
        for ytick in yticks:
            lats = np.full_like(lons, ytick)
            lat_line = shapely.LineString(np.c_[lons, lats])
            lat_line = ax.projection.project_geometry(lat_line, PLATE_CARREE)
            pointL = lineL.intersection(lat_line)
            if isinstance(pointL, shapely.Point) and not pointL.is_empty:
                yticksL.append(pointL.y)
                yticklabelsL.append(yformatter(ytick))
            pointR = lineR.intersection(lat_line)
            if isinstance(pointR, shapely.Point) and not pointR.is_empty:
                yticksR.append(pointR.y)
                yticklabelsR.append(yformatter(ytick))

        return yticksL, yticksR, yticklabelsL, yticklabelsR

    # 通过隐藏部分刻度，实现上下刻度不同的效果。
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

    # 通过 alpha 隐藏部分刻度，实现左右刻度不同的效果。
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


def _interp_minor_ticks(major_ticks: NDArray, m: int) -> NDArray:
    """在主刻度的每段间隔内线性插值出 m 个次刻度"""
    n = len(major_ticks)
    if n <= 1 or m == 0:
        return np.array([])

    L = n + (n - 1) * m
    x = np.array([i for i in range(L) if i % (m + 1)]) / (L - 1)
    xp = np.linspace(0, 1, n)
    major_ticks = np.sort(major_ticks)
    minor_ticks = np.interp(x, xp, major_ticks)

    return minor_ticks


def set_map_ticks(
    ax: Axes,
    extents: Sequence[float] | None = None,
    xticks: ArrayLike | None = None,
    yticks: ArrayLike | None = None,
    *,
    dx: float = 10,
    dy: float = 10,
    mx: int = 0,
    my: int = 0,
    xformatter: Formatter | None = None,
    yformatter: Formatter | None = None,
) -> None:
    """
    设置地图的范围和刻度

    当 ax 是普通 Axes 时，认为其投影是 PlateCarree。
    当 ax 是 GeoAxes 时，如果 ax 的边框不是矩形或跨越了日界线，可能产生错误的结果。
    此时建议换用 GeoAxes.gridlines。

    Parameters
    ----------
    ax : Axes
        目标 Axes

    extents : (4,) tuple of float, default None
        经纬度范围 (lon0, lon1, lat0, lat1)。默认为 None，表示显示全球。
        当 GeoAxes 的投影不是 PlateCarree 或 Mercator 时 extents 不能为 None。

    xticks : array_like, default None
        x 轴主刻度的坐标，单位为经度。
        默认为 None，表示不直接给出，而是由 dx 自动决定。

    yticks : array_like, default None
        y 轴主刻度的坐标，单位为纬度。
        默认为 None，表示不直接给出，而是由 dy 自动决定。

    dx : float, default 10
        以 dx 为间隔从 -180 度开始生成 x 轴主刻度的间隔。默认为 10 度。
        xticks 不为 None 时该参数无效。

    dy : float, default 10
        以 dy 为间隔从 -90 度开始生成 y 轴主刻度的间隔。默认为 10 度。
        yticks 不为 None 时该参数无效。

    mx : int, default 0
        经度主刻度之间次刻度的个数。默认为 0。

    my : int, default 0
        纬度主刻度之间次刻度的个数。默认为 0。

    xformatter : Formatter, default None
        x 轴刻度标签的 Formatter。默认为 None，表示 LongitudeFormatter。

    yformatter : Formatter, default None
        y 轴刻度标签的 Formatter。默认为 None，表示 LatitudeFormatter。
    """
    if extents is not None:
        lon0, lon1, lat0, lat1 = extents
        if lon0 >= lon1 or lat0 >= lat1:
            raise ValueError("要求 lon0 < lon1 且 lat0 < lat1")

    if dx < 0 or dy < 0:
        raise ValueError("dx 和 dy 必须是正数")

    if xticks is None:
        major_xticks = np.arange(math.floor(360 / dx) + 1) * dx - 180
    else:
        major_xticks = np.asarray(xticks)

    if yticks is None:
        major_yticks = np.arange(math.floor(180 / dy) + 1) * dy - 90
    else:
        major_yticks = np.asarray(yticks)

    if not isinstance(mx, int) or mx < 0:
        raise ValueError("mx 只能是非负整数")
    minor_xticks = _interp_minor_ticks(major_xticks, mx)

    if not isinstance(my, int) or my < 0:
        raise ValueError("my 只能是非负整数")
    minor_yticks = _interp_minor_ticks(major_yticks, my)

    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()

    kwargs = {
        "ax": ax,
        "extents": extents,
        "major_xticks": major_xticks,
        "major_yticks": major_yticks,
        "minor_xticks": minor_xticks,
        "minor_yticks": minor_yticks,
        "xformatter": xformatter,
        "yformatter": yformatter,
    }

    match ax:
        case GeoAxes():
            if isinstance(ax.projection, (PlateCarree, Mercator)):
                _set_simple_geoaxes_ticks(**kwargs)
            else:
                _set_complex_geoaxes_ticks(**kwargs)
        case Axes():
            _set_axes_ticks(**kwargs)
        case _:
            raise TypeError(format_type_error("ax", ax, Axes))


def quick_cn_map(
    extents: Sequence[float] = (70, 140, 0, 60),
    use_geoaxes: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Axes:
    """
    快速制作带省界和九段线的中国地图

    Parameters
    ----------
    extents : (4,) tuple of float, default (70, 140, 0, 60)
        经纬度范围 (lon0, lon1, lat0, lat1)。默认为 (70, 140, 0, 60)。

    use_geoaxes : bool, default True
        是否使用 GeoAxes。默认为 True。

    figsize : (2,) tuple of int, default None
        Figure 的宽高。默认为 None，表示 (6.4, 4.8)。

    Returns
    -------
    ax : Axes
        表示地图的 Axes
    """
    fig = plt.figure(figsize=figsize)
    if use_geoaxes:
        ax = fig.add_subplot(projection=PLATE_CARREE)
    else:
        ax = fig.add_subplot()
        ax.set_aspect(1)

    set_map_ticks(ax, extents)
    add_cn_province(ax)
    add_cn_line(ax)

    return ax


def add_quiver_legend(
    Q: Quiver,
    U: float,
    units: str = "m/s",
    width: float = 0.15,
    height: float = 0.15,
    loc: Literal[
        "lower left", "lower right", "upper left", "upper right"
    ] = "lower right",
    qk_kwargs: dict | None = None,
    patch_kwargs: dict | None = None,
) -> QuiverLegend:
    """
    在 Axes 的角落添加 Quiver 的图例（带矩形背景的 QuiverKey）

    箭头下方有形如 '{U} {units}' 的标签。

    Parameters
    ----------
    Q : Quiver
        Axes.quiver 返回的对象

    U : float
        箭头长度

    units : str, default "m/s"
        标签单位。默认为 m/s。

    width : float, default 0.15
        图例宽度。基于 Axes 坐标，默认为 0.15。

    height : float, default 0.15
        图例高度。基于 Axes 坐标，默认为 0.15。

    loc : {'lower left', 'lower right', 'upper left', 'upper right'}, default 'lower right'
        图例位置。默认为 'lower right'。

    qk_kwargs : dict, default None
        QuiverKey 类的关键字参数。默认为 None。
        例如 labelsep、labelcolor、fontproperties 等。
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiverkey.html

    patch_kwargs : dict, default None
        表示背景方框的 Rectangle 类的关键字参数。默认为 None。
        例如 linewidth、edgecolor、facecolor 等。
        https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html

    Returns
    -------
    quiver_legend : QuiverLegend
        图例对象
    """
    quiver_legend = QuiverLegend(
        Q=Q,
        U=U,
        units=units,
        width=width,
        height=height,
        loc=loc,
        qk_kwargs=qk_kwargs,
        patch_kwargs=patch_kwargs,
    )
    ax = cast(Axes, Q.axes)
    ax.add_artist(quiver_legend)

    return quiver_legend


def add_compass(
    ax: Axes,
    x: float,
    y: float,
    angle: float | None = None,
    size: float = 20,
    style: Literal["arrow", "star", "circle"] = "arrow",
    pc_kwargs: dict | None = None,
    text_kwargs: dict | None = None,
) -> Compass:
    """
    在 Axes 上添加指北针

    Parameters
    ----------
    ax : Axes
        目标 Axes

    x, y : float
        指北针的横纵坐标。基于 Axes 坐标系。

    angle : float, default None
        指北针的方位角。单位为度。
        默认为 None。表示 GeoAxes 会自动计算角度，而 Axes 默认 0 度（正北）。

    size : float, default 20
        指北针大小。单位为点，默认为 20。

    style : {'arrow', 'circle', 'star'}, default 'arrow'
        指北针造型。默认为 'arrow'。

    pc_kwargs : dict, default None
        表示指北针的 PathCollection 类的关键字参数。默认为 None。
        例如 linewidth、edgecolor、facecolor 等。
        https://matplotlib.org/stable/api/collections_api.html

    text_kwargs : dict, default None
        表示指北针 N 字的 Text 类的关键字参数。默认为 None。
        https://matplotlib.org/stable/api/text_api.html

    Returns
    -------
    compass : Compass
        指北针对象
    """
    if not isinstance(ax, Axes):
        raise TypeError(format_type_error("ax", ax, Axes))

    compass = Compass(
        x=x,
        y=y,
        angle=angle,
        size=size,
        style=style,
        pc_kwargs=pc_kwargs,
        text_kwargs=text_kwargs,
    )
    ax.add_artist(compass)

    return compass


def add_scale_bar(
    ax: Axes,
    x: float,
    y: float,
    length: float = 1000,
    units: Literal["m", "km"] = "km",
) -> ScaleBar:
    """
    在 Axes 上添加地图比例尺

    当 ax 是普通 Axes 时，认为其投影是 PlateCarree。
    会根据 ax 的投影计算比例尺大小。

    Parameters
    ----------
    ax : Axes
        目标 Axes

    x, y : float
        比例尺左端的横纵坐标。基于 Axes 坐标系。

    length : float, default 1000
        比例尺长度。默认为 1000。

    units : {'m', 'km'}, default 'km'
        比例尺长度的单位。默认为 'km'。

    Returns
    -------
    scale_bar : ScaleBar
        比例尺对象。刻度可以通过 set_xticks、tick_params 等方法修改。
    """
    return ScaleBar(ax, x, y, length, units)


def add_frame(ax: Axes, width: float = 5, **kwargs: Any) -> Frame:
    """
    在 Axes 上添加 GMT 风格的边框

    需要先设置好 Axes 的刻度，再调用该函数。

    Parameters
    ----------
    ax : Axes
        目标 Axes。仅支持 PlateCarree 和 Mercator 投影的 GeoAxes。
        若 ax 是 ScaleBar, 只会添加 top 边框。

    width : float, default 5
        边框的宽度。单位为点，默认为 5。

    **kwargs
        表示边框的 PathCollection 类的关键字参数。
        例如 linewidth、edgecolor、facecolor 等。
        https://matplotlib.org/stable/api/collections_api.html

    Returns
    -------
    frame : Frame
        边框对象
    """
    if not isinstance(ax, Axes):
        raise TypeError(format_type_error("ax", ax, Axes))

    frame = Frame(width, **kwargs)
    ax.add_artist(frame)

    return frame


def add_box(
    ax: Axes, extents: Sequence[float], steps: int = 100, **kwargs: Any
) -> PathPatch:
    """
    在 Axes 上添加一个方框

    Parameters
    ----------
    ax : Axes
        目标 Axes

    extents : (4,) tuple of float
        方框范围 (x0, x1, y0, y1)

    steps: int, default 100
        在方框上重采样出 N * steps 个点。默认为 100。
        当 ax 是 GeoAxes 且指定 transform 关键字时能保证方框的平滑。

    **kwargs
        PathPatch 类的关键字参数
        例如 linewidth、edgecolor、facecolor 和 transform 等。
        https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.PathPatch.html

    Returns
    -------
    patch : PathPatch
        方框对象
    """
    if not isinstance(ax, Axes):
        raise TypeError(format_type_error("ax", ax, Axes))

    # 设置参数
    kwargs = normalize_kwargs(kwargs, PathPatch)
    kwargs.setdefault("edgecolor", "r")
    kwargs.setdefault("facecolor", "none")

    # 添加 Patch
    path = box_path(*extents).interpolated(steps)  # type: ignore
    patch = PathPatch(path, **kwargs)
    ax.add_patch(patch)

    return patch


def add_mini_axes(
    ax: Axes,
    shrink: float = 0.4,
    aspect: float = 1,
    loc: Literal[
        "lower left", "lower right", "upper left", "upper right"
    ] = "lower right",
    projection: Literal["same", None] | CRS = "same",
) -> Axes:
    """
    在 Axes 的角落添加一个新的 Axes 并返回

    Parameters
    ----------
    ax : Axes
        目标 Axes

    shrink : float, default 0.4
        缩小倍数。默认为 0.4。shrink=1 时新 Axes 与目标 Axes 等高或等宽。

    aspect : float, default 1
        单位坐标的高宽比。默认为 1，与 GeoAxes 相同。

    loc : {'lower left', 'lower right', 'upper left', 'upper right'}, default 'lower right'
        指定放置在哪个角落。默认为 'lower right'。

    projection : {'same', None} or CRS, default 'same'
        新 Axes 的投影。默认为 'same'，表示沿用 ax 的投影。
        如果是 None，表示没有投影。

    Returns
    -------
    Axes
        新的 Axes
    """
    if not isinstance(ax, Axes):
        raise TypeError(format_type_error("ax", ax, Axes))

    if projection == "same":
        projection = getattr(ax, "projection", None)
    new_ax = ax.figure.add_subplot(projection=projection)  # type: ignore
    new_ax.set_aspect(aspect)
    draw = new_ax.draw

    @wraps(draw)
    def wrapper(renderer: RendererBase) -> None:
        """在保持宽高比的前提下将 new_ax 缩小到 ax 的角落"""
        bbox = ax.get_position()
        new_bbox = new_ax.get_position()
        if bbox.width / bbox.height < new_bbox.width / new_bbox.height:
            ratio = bbox.width / new_bbox.width * shrink
        else:
            ratio = bbox.height / new_bbox.height * shrink
        width = new_bbox.width * ratio
        height = new_bbox.height * ratio

        if loc == "lower left":
            x0 = bbox.x0
            x1 = bbox.x0 + width
            y0 = bbox.y0
            y1 = bbox.y0 + height
        elif loc == "lower right":
            x0 = bbox.x1 - width
            x1 = bbox.x1
            y0 = bbox.y0
            y1 = bbox.y0 + height
        elif loc == "upper left":
            x0 = bbox.x0
            x1 = bbox.x0 + width
            y0 = bbox.y1 - height
            y1 = bbox.y1
        elif loc == "upper right":
            x0 = bbox.x1 - width
            x1 = bbox.x1
            y0 = bbox.y1 - height
            y1 = bbox.y1
        else:
            raise ValueError(
                format_literal_error(
                    "loc",
                    loc,
                    {"lower left", "lower right", "upper left", "upper right"},
                )
            )

        new_bbox = Bbox.from_extents(x0, y0, x1, y1)
        new_ax.set_position(new_bbox)
        draw(renderer)

    new_ax.draw = wrapper

    return new_ax


def get_cross_section_xticks(
    lon: ArrayLike,
    lat: ArrayLike,
    nticks: int = 6,
    lon_formatter: Formatter | None = None,
    lat_formatter: Formatter | None = None,
) -> tuple[NDArray, NDArray, list[str]]:
    """
    返回垂直截面图所需的横坐标，刻度位置和刻度标签。

    用经纬度的欧式距离表示横坐标，在横坐标上取 nticks 个等距的刻度，
    利用线性插值计算每个刻度对应的经纬度值并用作刻度标签。

    Parameters
    ----------
    lon, lat: (npts,) array_like
        横截面对应的经纬度数组

    nticks : int, default 6
        刻度的数量。默认为 6。

    lon_formatter : Formatter, default None
        刻度标签里经度的 Formatter，用来控制字符串的格式。
        默认为 None，表示 LongitudeFormatter。

    lat_formatter : Formatter, default None
        刻度标签里纬度的 Formatter。用来控制字符串的格式。
        默认为 None，表示 LatitudeFormatter。

    Returns
    -------
    x : (npts,) ndarray
        横截面的横坐标

    xticks : (nticks,) ndarray
        刻度的横坐标

    xticklabels : (nticks,) list of str
        用经纬度表示的刻度标签
    """
    # 线性插值计算刻度的经纬度值
    lon, lat = asarrays(lon, lat)
    if lon.ndim != 1:
        raise ValueError("lon 必须是一维数组")
    if len(lon) <= 1:
        raise ValueError("lon 至少有 2 个元素")
    if lon.shape != lat.shape:
        raise ValueError("lon 和 lat 的长度必须相同")

    dlon = lon - lon[0]
    dlat = lat - lat[0]
    x = np.hypot(dlon, dlat)
    xticks = np.linspace(x[0], x[-1], nticks)
    tlon = np.interp(xticks, x, lon)
    tlat = np.interp(xticks, x, lat)

    # 获取字符串形式的刻度标签
    xticklabels = []
    if lon_formatter is None:
        lon_formatter = LongitudeFormatter(number_format=".1f")
    if lat_formatter is None:
        lat_formatter = LatitudeFormatter(number_format=".1f")
    for i in range(nticks):
        lon_label = lon_formatter(tlon[i])
        lat_label = lat_formatter(tlat[i])
        xticklabels.append(lon_label + "\n" + lat_label)

    return x, xticks, xticklabels


def make_qualitative_palette(
    colors: list | NDArray,
) -> tuple[ListedColormap, Normalize, NDArray]:
    """
    创建一组定性的 colormap 和 norm，同时返回刻度位置。

    Parameters
    ----------
    colors : (N,) list or (N, 3) or (N, 4) array_like
        colormap 所含的颜色。可以为含有颜色的序列或 RGB(A) 数组。

    Returns
    -------
    cmap : ListedColormap
        创建的 colormap

    norm : Normalize
        创建的 norm。N 个颜色对应于 0~N-1 范围的数据。

    ticks : (N,) ndarray
        colorbar 刻度的坐标
    """
    N = len(colors)
    cmap = ListedColormap(colors)
    norm = Normalize(vmin=-0.5, vmax=N - 0.5)
    ticks = np.arange(N)

    return cmap, norm, ticks


def get_aod_cmap() -> ListedColormap:
    """返回适用于 AOD 的 cmap"""
    filepath = DATA_DIRPATH / "NEO_modis_aer_od.csv"
    rgb = np.loadtxt(str(filepath), delimiter=",") / 256
    cmap = ListedColormap(rgb)

    return cmap


class CenteredBoundaryNorm(BoundaryNorm):
    """将 vcenter 所在的 bin 映射到 cmap 中间的 BoundaryNorm"""

    def __init__(
        self, boundaries: ArrayLike, vcenter: float = 0, clip: bool = False
    ) -> None:
        boundaries = np.asarray(boundaries)
        super().__init__(boundaries, len(boundaries) - 1, clip)
        self.N1 = np.count_nonzero(boundaries < vcenter)
        self.N2 = np.count_nonzero(boundaries > vcenter)
        if self.N1 < 1 or self.N2 < 1:
            raise ValueError("vcenter 两侧至少各有一条边界")

    def __call__(self, value: Any, clip: bool | None = None) -> np.ma.MaskedArray:  # type: ignore
        # 将 BoundaryNorm 的 [0, N-1] 映射到 [0.0, 1.0] 内
        result = super().__call__(value, clip)
        if self.N1 + self.N2 == self.N - 1:
            result = np.ma.where(
                result < self.N1,
                result / (2 * self.N1),
                (result - self.N1 + self.N2 + 1) / (2 * self.N2),
            )
        else:
            # MaskedArray 除以零不会报错
            result = np.ma.where(
                result < self.N1,
                result / (2 * (self.N1 - 1)),
                (result - self.N1 + self.N2) / (2 * (self.N2 - 1)),
            )

        return result


def plot_colormap(
    cmap: Colormap,
    norm: Normalize | None = None,
    extend: Literal["neither", "both", "min", "max"] | None = None,
    ax: Axes | None = None,
) -> Colorbar:
    """快速展示一条 colormap"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 1))
        fig.subplots_adjust(bottom=0.5)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(mappable, cax=ax, orientation="horizontal", extend=extend)

    return cbar


def letter_axes(
    axes: ArrayLike, x: ArrayLike, y: ArrayLike, **kwargs: Any
) -> list[Text]:
    """
    给一组 Axes 按顺序标注字母

    Parameters
    ----------
    axes : Axes or array_like of Axes
        Axes 的数组

    x, y: float or array_like of float
        字母的横纵坐标，基于 Axes 单位。
        可以为标量或数组，数组形状需与 axes 相同。

    **kwargs
        Text 类的关键字参数。
        例如 fontsize、fontfamily 和 color 等。
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
    """
    axes = np.atleast_1d(axes)
    x = np.full_like(axes, x) if np.isscalar(x) else np.asarray(x)
    y = np.full_like(axes, y) if np.isscalar(y) else np.asarray(y)

    texts = []
    for i, (ax, xi, yi) in enumerate(zip(axes.flat, x.flat, y.flat)):
        letter = chr(ord("a") + i)
        text = ax.text(
            x=xi,
            y=yi,
            s=f"({letter})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            **kwargs,
        )
        texts.append(text)

    return texts


def load_test_data() -> NpzFile:
    """读取测试用的数据。包含地表 2m 气温（K）和水平 10m 风速。"""
    filepath = DATA_DIRPATH / "test.npz"
    return np.load(str(filepath))


def savefig(fname: Any, fig: Figure | None = None, **kwargs: Any) -> None:
    """保存 Figure 为图片"""
    if fig is None:
        fig = plt.gcf()
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(fname, **kwargs)


def get_font_names(sub: str | None = None) -> list[str]:
    """获取 Matplotlib 可用的字体名称"""
    names = font_manager.get_font_names()
    if sub is not None:
        return [name for name in names if sub.lower() in name.lower()]
    return names


@deprecator(alternative=add_geometries)
def add_geoms(
    ax: Axes,
    geoms: BaseGeometry | Sequence[BaseGeometry],
    crs: CRS | None = None,
    fast_transform: bool = True,
    skip_outside: bool = True,
    **kwargs: Any,
) -> GeometryPathCollection:
    return add_geometries(
        ax=ax,
        geometries=geoms,
        crs=crs,
        fast_transform=fast_transform,
        skip_outside=skip_outside,
        **kwargs,
    )


@deprecator(alternative=make_qualitative_palette)
def get_qualitative_palette(
    colors: list | NDArray,
) -> tuple[ListedColormap, Normalize, NDArray]:
    return make_qualitative_palette(colors)
