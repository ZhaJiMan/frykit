from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import partial, wraps
from threading import Lock
from typing import Any, Iterable, Literal, cast
from weakref import WeakKeyDictionary, WeakValueDictionary

import numpy as np
import shapely
from cartopy.crs import CRS, Mercator, PlateCarree, Projection
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.quiver import Quiver, QuiverKey
from matplotlib.text import Text
from matplotlib.transforms import Affine2D, Bbox, ScaledTranslation, offset_copy
from numpy import ma
from numpy.typing import ArrayLike, NDArray
from shapely.geometry.base import BaseGeometry
from typing_extensions import Unpack

from frykit.calc import get_values_between, t_to_az
from frykit.conf import config
from frykit.plot.projection import PLATE_CARREE
from frykit.plot.typing import (
    CompassPcKwargs,
    CompassTextKwargs,
    FrameKwargs,
    GeometryPathCollectionKwargs,
    QuiverLegendPatchKwargs,
    QuiverLegendQkKwargs,
    TextCollectionKwargs,
)
from frykit.plot.utils import (
    EMPTY_PATH,
    box_path,
    geometry_to_path,
    get_axes_extents,
    project_geometry,
)
from frykit.shp.typing import PolygonType
from frykit.typing import P, T
from frykit.utils import format_literal_error, format_type_error

__all__ = [
    "Compass",
    "Frame",
    "GeometryKey",
    "GeometryPathCollection",
    "QuiverLegend",
    "ScaleBar",
    "TextCollection",
    "clear_path_cache",
]

_lock = Lock()


def _with_lock(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with _lock:
            return func(*args, **kwargs)

    return wrapper


# https://github.com/SciTools/cartopy/blob/main/lib/cartopy/mpl/feature_artist.py
class GeometryKey:
    """用几何对象的 id 作为 key"""

    def __init__(self, geometry: BaseGeometry) -> None:
        self._id = id(geometry)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GeometryKey):
            return self._id == other._id
        else:
            return False

    def __hash__(self) -> int:
        return hash(self._id)


# 相比直接用几何对象当缓存字典的 key，用 id 当作弱引用的 key 要快得多
_key_to_geometry: WeakValueDictionary[GeometryKey, BaseGeometry] = WeakValueDictionary()
_key_to_crs_to_path: WeakKeyDictionary[
    GeometryKey, dict[tuple[CRS, bool] | None, Path]
] = WeakKeyDictionary()


@_with_lock
def clear_path_cache() -> None:
    """清理几何对象到 Path 的缓存"""
    _key_to_geometry.clear()
    _key_to_crs_to_path.clear()


@_with_lock
def _cached_geometry_to_path(geometry: BaseGeometry) -> Path:
    key1 = GeometryKey(geometry)
    key2 = None
    _key_to_geometry.setdefault(key1, geometry)
    crs_to_path = _key_to_crs_to_path.setdefault(key1, {})
    path = crs_to_path.get(key2)
    if path is None:
        path = geometry_to_path(geometry)
        crs_to_path[key2] = path

    return path


def _cached_geometries_to_paths(
    geometries: Iterable[BaseGeometry] | NDArray[np.object_],
) -> list[Path]:
    return list(map(_cached_geometry_to_path, geometries))


@_with_lock
def _cached_project_geometries_to_paths(
    geometries: Sequence[BaseGeometry] | NDArray[np.object_],
    crs_from: CRS,
    crs_to: Projection,
    fast_transform: bool = True,
) -> list[Path]:
    geometries = np.asarray(geometries, dtype=np.object_)
    paths = np.empty_like(geometries)
    miss_mask = np.ones_like(geometries, dtype=bool)
    miss_maps: list[dict[tuple[CRS, bool] | None, Path]] = []

    key2 = (crs_to, fast_transform)
    for i, geometry in enumerate(geometries):
        key1 = GeometryKey(geometry)
        _key_to_geometry.setdefault(key1, geometry)
        crs_to_path = _key_to_crs_to_path.setdefault(key1, {})
        path = crs_to_path.get(key2)
        if path is not None:
            paths[i] = path
            miss_mask[i] = False
        else:
            miss_maps.append(crs_to_path)

    if not miss_maps:
        return paths.tolist()

    if fast_transform:
        # 批量投影效率略高一些
        miss_geometries = project_geometry(geometries[miss_mask], crs_from, crs_to)
    else:
        miss_geometries = [
            crs_to.project_geometry(geometry, crs_from)
            for geometry in geometries[miss_mask]
        ]

    miss_paths = list(map(geometry_to_path, miss_geometries))
    paths[miss_mask] = miss_paths
    for crs_to_path, path in zip(miss_maps, miss_paths):
        crs_to_path[key2] = path

    return paths.tolist()


def _cached_project_geometry_to_path(
    geometry: BaseGeometry,
    crs_from: CRS,
    crs_to: Projection,
    fast_transform: bool = True,
) -> Path:
    return _cached_project_geometries_to_paths(
        [geometry], crs_from, crs_to, fast_transform
    )[0]


def _resolve_fast_transform(fast_transform: bool | None) -> bool:
    if fast_transform is None:
        return config.fast_transform
    else:
        config.validate("fast_transform", fast_transform)
        return fast_transform


def _resolve_skip_outside(skip_outside: bool | None) -> bool:
    if skip_outside is None:
        return config.skip_outside
    else:
        config.validate("skip_outside", skip_outside)
        return skip_outside


# TODO: 类持有对几何对象的引用，会存在问题吗？
class GeometryPathCollection(PathCollection):
    """投影并绘制几何对象的 PathCollection 类"""

    def __init__(
        self,
        ax: Axes,
        geometries: Sequence[BaseGeometry] | NDArray[np.object_],
        crs: CRS | None,
        fast_transform: bool | None = None,
        skip_outside: bool | None = None,
        **kwargs: Unpack[GeometryPathCollectionKwargs],
    ) -> None:
        self.geometries = np.asarray(geometries, dtype=np.object_)
        self.fast_transform = _resolve_fast_transform(fast_transform)
        self.skip_outside = _resolve_skip_outside(skip_outside)

        match ax:
            case GeoAxes():
                self.crs = PLATE_CARREE if crs is None else crs
                self._geometries_to_paths = partial(
                    _cached_project_geometries_to_paths,
                    crs_from=self.crs,
                    crs_to=ax.projection,
                    fast_transform=self.fast_transform,
                )
            case Axes():
                if crs is not None:
                    raise ValueError("ax 不是 GeoAxes 时 crs 只能为 None")
                self.crs = None
                self._geometries_to_paths = _cached_geometries_to_paths
            case _:
                raise TypeError(format_type_error("ax", ax, Axes))

        # 用投影后的 bbox 做初始化的 path
        paths: list[Path] = []
        if ax.get_autoscale_on() and len(geometries) > 0:
            bounds = shapely.bounds(geometries)
            bounds = ma.masked_invalid(bounds)
            x0, y0 = bounds[:, :2].min(axis=0)
            x1, y1 = bounds[:, 2:].max(axis=0)
            if all(x is not ma.masked for x in [x0, y0, x1, y1]):
                path = box_path(x0, x1, y0, y1).interpolated(100)
                polygon = shapely.Polygon(path.vertices)  # pyright: ignore[reportArgumentType]
                paths = self._geometries_to_paths([polygon])

        super().__init__(paths, **kwargs)
        ax.add_collection(self)
        ax._request_autoscale_view()  # pyright: ignore[reportAttributeAccessIssue]

    def _init(self) -> None:
        if not self.skip_outside:
            paths = self._geometries_to_paths(self.geometries)
            self.set_paths(paths)
            return

        # 只投影和绘制可见的几何对象
        ax = cast(Axes, self.axes)
        if isinstance(ax, GeoAxes):
            x0, x1, y0, y1 = ax.get_extent(self.crs)
        else:
            x0, x1, y0, y1 = get_axes_extents(ax)
        box = shapely.box(x0, y0, x1, y1)
        shapely.prepare(box)
        mask = box.intersects(self.geometries)
        paths = np.full_like(self.geometries, EMPTY_PATH)
        paths[mask] = self._geometries_to_paths(self.geometries[mask])
        self.set_paths(paths.tolist())

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        if self.get_visible():
            self._init()
            super().draw(renderer)


class TextCollection(Artist):
    """绘制一组文本的类"""

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        s: ArrayLike,
        skip_outside: bool | None = None,
        **kwargs: Unpack[TextCollectionKwargs],
    ) -> None:
        super().__init__()
        self.skip_outside = _resolve_skip_outside(skip_outside)
        self.set_zorder(kwargs.get("zorder", 3))
        if "transform" in kwargs:
            self.set_transform(kwargs["transform"])

        self.x, self.y, self.s = map(np.atleast_1d, [x, y, s])
        if self.x.ndim != 1:
            raise ValueError("x 必须是一维数组")
        if not self.x.shape == self.y.shape == self.s.shape:
            raise ValueError("x、y 和 s 长度必须相同")
        self.coords = np.column_stack([self.x, self.y])

        kwargs = cast(TextCollectionKwargs, normalize_kwargs(kwargs, Text))
        kwargs.setdefault("horizontalalignment", "center")
        kwargs.setdefault("verticalalignment", "center")
        kwargs.setdefault("clip_on", True)
        self.texts = [
            Text(xi, yi, si, **kwargs)
            for xi, yi, si in zip(*map(np.ndarray.tolist, [self.x, self.y, self.s]))
        ]

    def set_figure(self, fig: Figure | SubFigure) -> None:
        super().set_figure(fig)
        for text in self.texts:
            text.set_figure(fig)

    def _clip_by_polygon(self, polygon: PolygonType) -> None:
        # 要求 polygon 基于 data 坐标系
        if self.axes is None:
            raise ValueError("必须设置 TextCollection.axes")

        # 只绘制可见的文本对象
        shapely.prepare(polygon)
        trans = self.get_transform() - self.axes.transData
        coords = trans.transform(self.coords)
        mask = shapely.contains_xy(polygon, coords[:, 0], coords[:, 1])
        for i in np.nonzero(~mask)[0]:
            self.texts[i].set_visible(False)

    def _init(self) -> None:
        if self.axes is not None and self.skip_outside:
            x0, x1, y0, y1 = get_axes_extents(self.axes)
            box = shapely.box(x0, y0, x1, y1)
            self._clip_by_polygon(box)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        if self.get_visible():
            self._init()
            for text in self.texts:
                text.draw(renderer)
            self.stale = False


class QuiverLegend(QuiverKey):
    """Quiver 图例"""

    def __init__(
        self,
        Q: Quiver,
        U: float,
        units: str = "m/s",
        width: float = 0.15,
        height: float = 0.15,
        loc: Literal[
            "lower left", "lower right", "upper left", "upper right"
        ] = "lower right",
        qk_kwargs: QuiverLegendQkKwargs | None = None,
        patch_kwargs: QuiverLegendPatchKwargs | None = None,
    ) -> None:
        self.units = units
        self.width = width
        self.height = height
        self.loc = loc

        match loc:
            case "lower left":
                X = width / 2
                Y = height / 2
            case "lower right":
                X = 1 - width / 2
                Y = height / 2
            case "upper left":
                X = width / 2
                Y = 1 - height / 2
            case "upper right":
                X = 1 - width / 2
                Y = 1 - height / 2
            case _:
                raise ValueError(
                    format_literal_error(
                        "loc",
                        loc,
                        ["lower left", "lower right", "upper left", "upper right"],
                    )
                )

        if qk_kwargs is None:
            qk_kwargs = {}

        super().__init__(
            Q=Q,
            X=X,
            Y=Y,
            U=U,
            label=f"{U} {units}",
            labelpos="S",
            coordinates="axes",
            **qk_kwargs,
        )

        # zorder 必须在初始化后设置
        zorder = qk_kwargs.get("zorder", 5)
        self.set_zorder(zorder)

        if patch_kwargs is None:
            patch_kwargs = {}
        patch_kwargs = cast(
            QuiverLegendPatchKwargs, normalize_kwargs(patch_kwargs, Rectangle)
        )
        patch_kwargs.setdefault("linewidth", 0.8)
        patch_kwargs.setdefault("edgecolor", "k")
        patch_kwargs.setdefault("facecolor", "w")

        self.patch = Rectangle(
            xy=(X - width / 2, Y - height / 2),
            width=width,
            height=height,
            transform=None,
            **patch_kwargs,
        )

        if Q.axes is None:
            raise ValueError("必须设置 Q.axes")
        if Q.axes.figure is None:
            raise ValueError("必须设置 Q.axes.figure")
        ax = Q.axes
        fig = Q.axes.figure

        # 将 qk 调整至 patch 的中心
        self._labelsep_inches: float
        fontsize = cast(int, self.text.get_fontsize()) / 72
        dy = (self._labelsep_inches + fontsize) / 2
        trans = offset_copy(ax.transAxes, fig, 0, dy)  # pyright: ignore[reportArgumentType]
        self.set_transform(trans)

        self.patch.axes = ax
        self.patch.set_transform(ax.transAxes)
        ax.add_artist(self)

    def _set_transform(self) -> None:
        pass  # 无效化 QuiveKey 的同名方法

    def set_figure(self, fig: Figure | SubFigure) -> None:
        self.patch.set_figure(fig)
        super().set_figure(fig)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        if self.get_visible():
            self.patch.draw(renderer)
            super().draw(renderer)


# TODO: Geodetic 配合 Mercator.GOOGLE 时速度很慢？
class Compass(PathCollection):
    """地图指北针"""

    def __init__(
        self,
        x: float,
        y: float,
        angle: float | None = None,
        size: float = 20,
        style: Literal["arrow", "star", "circle"] = "arrow",
        pc_kwargs: CompassPcKwargs | None = None,
        text_kwargs: CompassTextKwargs | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.angle = angle
        self.size = size
        self.style = style

        head = size / 72
        match style:
            case "arrow":
                width = axis = head * 2 / 3
                paths = self._make_paths(width, head, axis)
                colors = ["k", "w"]
            case "circle":
                width = axis = head * 2 / 3
                radius = head * 2 / 5
                paths = [
                    Path.circle((0, head / 9), radius),
                    *self._make_paths(width, head, axis),
                ]
                colors = ["none", "k", "w"]
            case "star":
                width = head / 3
                axis = head + width / 2
                paths = self._make_paths(width, head, axis)
                for deg in range(90, 360, 90):
                    rotation = Affine2D().rotate_deg(deg)
                    paths.append(paths[0].transformed(rotation))
                    paths.append(paths[1].transformed(rotation))
                colors = ["k", "w"]
            case _:
                raise ValueError(
                    format_literal_error("style", style, ["arrow", "circle", "star"])
                )

        if pc_kwargs is None:
            pc_kwargs = {}
        pc_kwargs = cast(CompassPcKwargs, normalize_kwargs(pc_kwargs, PathCollection))
        pc_kwargs.setdefault("linewidth", 1)
        pc_kwargs.setdefault("edgecolor", "k")
        pc_kwargs.setdefault("facecolor", colors)
        pc_kwargs.setdefault("clip_on", False)
        pc_kwargs.setdefault("zorder", 5)
        super().__init__(paths, transform=None, **pc_kwargs)

        # 文字在箭头上方
        if text_kwargs is None:
            text_kwargs = {}
        text_kwargs = cast(CompassTextKwargs, normalize_kwargs(text_kwargs, Text))
        text_kwargs.setdefault("fontsize", size / 1.5)
        pad = head / 2.5
        self.text = Text(
            x=0,
            y=axis + pad,
            text="N",
            ha="center",
            va="center",
            rotation=0,
            transform=None,
            **text_kwargs,
        )

    @staticmethod
    def _make_paths(width: float, head: float, axis: float) -> list[Path]:
        # 箭头方向朝上
        # width: 箭头宽度
        # head: 箭头长度
        # axis: 箭头中轴长度
        return [
            Path([(0, 0), (0, axis), (-width / 2, axis - head), (0, 0)]),
            Path([(0, 0), (width / 2, axis - head), (0, axis), (0, 0)]),
        ]

    def _init(self) -> None:
        if self.axes is None:
            raise ValueError("必须设置 Compass.axes")
        if self.axes.figure is None:
            raise ValueError("必须设置 Compass.axes.figure")
        ax = self.axes
        fig = self.axes.figure

        if self.angle is not None:
            azimuth = self.angle
        else:
            # 计算指北针的方向
            if isinstance(ax, GeoAxes):
                axes_to_data = ax.transAxes - ax.transData
                x0, y0 = axes_to_data.transform((self.x, self.y))
                lon0, lat0 = PLATE_CARREE.transform_point(x0, y0, ax.projection)
                lon1, lat1 = lon0, min(lat0 + 0.01, 90)
                x1, y1 = ax.projection.transform_point(lon1, lat1, PLATE_CARREE)
                theta = math.degrees(math.atan2(y1 - y0, x1 - x0))
                azimuth = float(t_to_az(theta, degrees=True))
            else:
                azimuth = 0

        rotation = Affine2D().rotate_deg(-azimuth)
        translation = ScaledTranslation(self.x, self.y, ax.transAxes)  # pyright: ignore[reportArgumentType]
        trans = fig.dpi_scale_trans + rotation + translation
        self.text.set_transform(trans)
        self.text.set_rotation(-azimuth)
        self.set_transform(trans)

    def set_figure(self, fig: Figure | SubFigure) -> None:
        self.text.set_figure(fig)
        super().set_figure(fig)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        if self.get_visible():
            self._init()
            self.text.draw(renderer)
            super().draw(renderer)


class ScaleBar(Axes):
    """地图比例尺"""

    def __init__(
        self,
        ax: Axes,
        x: float,
        y: float,
        length: float = 1000,
        units: Literal["m", "km"] = "km",
    ) -> None:
        if not isinstance(ax, Axes):
            raise TypeError(format_type_error("ax", ax, Axes))

        self.x = x
        self.y = y
        self.length = length
        self.units = units

        match units:
            case "m":
                self._units = 1
            case "km":
                self._units = 1000
            case _:
                raise ValueError(format_literal_error("units", units, ["m", "km"]))

        if ax.figure is None:
            raise ValueError("必须设置 ax.figure")
        super().__init__(ax.figure, (0, 0, 1, 1), zorder=5)  # pyright: ignore[reportArgumentType]
        ax.add_child_axes(self)

        # 只显示上边框的刻度
        self.set_xlabel(units, fontsize="medium")
        self.set_xlim(0, length)
        self.tick_params(
            which="both",
            left=False,
            labelleft=False,
            bottom=False,
            labelbottom=False,
            top=True,
            labeltop=True,
            labelsize="small",
        )

    def _init(self) -> None:
        ax = cast(Axes, self.axes)
        fig = cast(Figure, ax.figure)

        if isinstance(ax, GeoAxes):
            # 在 GeoAxes 中心取一段横线，计算单位长度对应的地理长度。
            geod = PLATE_CARREE.get_geod()
            x0, x1, y0, y1 = get_axes_extents(ax)
            xm = (x0 + x1) / 2
            ym = (y0 + y1) / 2
            dx = (x1 - x0) / 10
            x0_ = xm - dx / 2
            x1_ = xm + dx / 2
            lon0, lat0 = PLATE_CARREE.transform_point(x0_, ym, ax.projection)
            lon1, lat1 = PLATE_CARREE.transform_point(x1_, ym, ax.projection)
            dr = geod.inv(lon0, lat0, lon1, lat1)[2] / self._units
            dxdr = dx / dr
        else:
            # 普通 Axes 认为是 PlateCarree 投影，取中心纬度。
            Re = 6371e3
            L = 2 * np.pi * Re / 360
            lat0, lat1 = ax.get_ylim()
            lat = (lat0 + lat1) / 2
            drdx = L * math.cos(math.radians(lat))
            dxdr = self._units / drdx

        # 重新设置比例尺的大小和位置
        axes_to_data = ax.transAxes - ax.transData
        data_to_figure = ax.transData - fig.transSubfigure
        x, y = axes_to_data.transform((self.x, self.y))
        width = self.length * dxdr
        bbox = Bbox.from_bounds(x, y, width, 1e-4 * width)
        bbox = data_to_figure.transform_bbox(bbox)
        self.set_position(bbox)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        if self.get_visible():
            self._init()
            super().draw(renderer)


# TODO: 非矩形边框
class Frame(Artist):
    """GMT风格的边框"""

    def __init__(self, width: float = 5, **kwargs: Unpack[FrameKwargs]) -> None:
        self.width = width
        self._width = width / 72
        super().__init__()
        self.set_zorder(2.5)

        kwargs = cast(FrameKwargs, normalize_kwargs(kwargs, PathCollection))
        kwargs.setdefault("linewidth", 1)
        kwargs.setdefault("edgecolor", "k")
        kwargs.setdefault("facecolor", ["k", "w"])

        self.pc_dict = {
            loc: PathCollection([], transform=None, **kwargs)
            for loc in ["left", "right", "top", "bottom", "corner"]
        }

    def _init(self) -> None:
        if self.axes is None:
            raise ValueError("必须设置 Frame.axes")
        if self.axes.figure is None:
            raise ValueError("必须设置 Frame.axes.figure")
        ax = self.axes
        fig = self.axes.figure

        if isinstance(ax, GeoAxes) and not isinstance(
            ax.projection, (PlateCarree, Mercator)
        ):
            raise ValueError("只支持 PlateCarree 和 Mercator 投影")

        # 将 inches 坐标系的 width 转为 axes 坐标系里的 dx 和 dy
        inches_to_axes = fig.dpi_scale_trans - ax.transAxes
        mtx = inches_to_axes.get_matrix()
        dx = self._width * mtx[0, 0]
        dy = self._width * mtx[1, 1]

        xtrans = ax.get_xaxis_transform()
        ytrans = ax.get_yaxis_transform()
        self.pc_dict["top"].set_transform(xtrans)
        self.pc_dict["bottom"].set_transform(xtrans)
        self.pc_dict["left"].set_transform(ytrans)
        self.pc_dict["right"].set_transform(ytrans)
        self.pc_dict["corner"].set_transform(ax.transAxes)

        # 收集 xlim 和 ylim 范围内所有刻度并去重
        major_xticks = ax.xaxis.get_majorticklocs()
        minor_xticks = ax.xaxis.get_minorticklocs()
        major_yticks = ax.yaxis.get_majorticklocs()
        minor_yticks = ax.yaxis.get_minorticklocs()
        x0, x1, y0, y1 = get_axes_extents(ax)
        xticks = np.array([x0, x1, *major_xticks, *minor_xticks])
        yticks = np.array([y0, y1, *major_yticks, *minor_yticks])
        xticks = get_values_between(xticks, x0, x1)
        yticks = get_values_between(yticks, y0, y1)
        # 通过 round 抵消 central_longitude 导致的浮点误差
        xticks = np.unique(xticks.round(9))
        yticks = np.unique(yticks.round(9))
        nx = len(xticks)
        ny = len(yticks)

        top_paths = [
            box_path(xticks[i], xticks[i + 1], 1, 1 + dy) for i in range(nx - 1)
        ]
        # 即便 xaxis 方向反转也维持边框的颜色顺序
        if ax.xaxis.get_inverted():
            top_paths.reverse()
        self.pc_dict["top"].set_paths(top_paths)

        # 比例尺对象只设置上边框
        if isinstance(ax, ScaleBar):
            return

        bottom_paths = [
            box_path(xticks[i], xticks[i + 1], -dy, 0) for i in range(nx - 1)
        ]
        if ax.xaxis.get_inverted():
            bottom_paths.reverse()
        self.pc_dict["bottom"].set_paths(bottom_paths)

        left_paths = [box_path(-dx, 0, yticks[i], yticks[i + 1]) for i in range(ny - 1)]
        if ax.yaxis.get_inverted():
            left_paths.reverse()
        self.pc_dict["left"].set_paths(left_paths)

        right_paths = [
            box_path(1, 1 + dx, yticks[i], yticks[i + 1]) for i in range(ny - 1)
        ]
        if ax.yaxis.get_inverted():
            right_paths.reverse()
        self.pc_dict["right"].set_paths(right_paths)

        # 单独画出四个角落的方块
        corner_paths = [
            box_path(-dx, 0, -dy, 0),
            box_path(1, 1 + dx, -dy, 0),
            box_path(-dx, 0, 1, 1 + dy),
            box_path(1, 1 + dx, 1, 1 + dy),
        ]
        self.pc_dict["corner"].set_paths(corner_paths)
        fc = self.pc_dict["top"].get_facecolor()[-1]
        self.pc_dict["corner"].set_facecolor(fc)  # pyright: ignore[reportArgumentType]

    def set_figure(self, fig: Figure | SubFigure) -> None:
        super().set_figure(fig)
        for pc in self.pc_dict.values():
            pc.set_figure(fig)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        if self.get_visible():
            self._init()
            for pc in self.pc_dict.values():
                pc.draw(renderer)
            self.stale = False


class CurlyQuiver:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
