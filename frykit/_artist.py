import math
from collections.abc import Collection, Iterable
from functools import partial
from typing import Any, Literal
from weakref import WeakKeyDictionary, WeakValueDictionary

import cartopy.crs as ccrs
import matplotlib.transforms as mtransforms
import numpy as np
import shapely.geometry as sgeom
from cartopy.mpl.feature_artist import _GeomKey
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.axes._axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.quiver import Quiver, QuiverKey
from matplotlib.text import Text
from numpy.typing import ArrayLike
from shapely.geometry.base import BaseGeometry
from shapely.vectorized import contains

import frykit.shp as fshp
from frykit.calc import make_ellipse, t_to_az

PLATE_CARREE = ccrs.PlateCarree()

# geom 的引用计数为零时弱引用会自动清理缓存
_key_to_geom = WeakValueDictionary()
_key_to_path = WeakKeyDictionary()
_key_to_crs_to_path = WeakKeyDictionary()

_key_to_geom: WeakValueDictionary[_GeomKey, BaseGeometry]
_key_to_path: WeakKeyDictionary[_GeomKey, Path | None]
_key_to_crs_to_path: WeakKeyDictionary[_GeomKey, dict[ccrs.CRS, tuple[bool, Path]]]


def _geoms_to_paths(geoms: Iterable[BaseGeometry]) -> list[Path]:
    """将一组几何对象转为 Path 并缓存结果"""
    paths = []
    for geom in geoms:
        key = _GeomKey(geom)
        # 同时设置三个弱引用字典，保证引用的是同一个 key 对象。
        _key_to_geom.setdefault(key, geom)
        _key_to_crs_to_path.setdefault(key, {})
        path = _key_to_path.setdefault(key)
        if path is None:
            path = fshp.geom_to_path(geom)
            _key_to_path[key] = path
        paths.append(path)

    return paths


def _transform_geoms_to_paths(
    geoms: Iterable[BaseGeometry],
    crs_from: ccrs.CRS,
    crs_to: ccrs.Projection,
    fast_transform: bool = True,
) -> list[Path]:
    if fast_transform:
        # 直接变换 vertices 会在 clip 环节产生麻烦
        transform_geom = fshp.GeometryTransformer(crs_from, crs_to)
    else:
        transform_geom = lambda x: crs_to.project_geometry(x, crs_from)  # noqa: E731

    paths = []
    for geom in geoms:
        key = _GeomKey(geom)
        _key_to_geom.setdefault(key, geom)
        _key_to_path.setdefault(key)
        mapping = _key_to_crs_to_path.setdefault(key, {})
        value = mapping.get(crs_to)
        if value is None or value[0] != fast_transform:
            geom = transform_geom(geom)
            path = fshp.geom_to_path(geom)
            value = (fast_transform, path)
            mapping[crs_to] = value
        paths.append(value[1])

    return paths


def _get_geoms_extents(
    geoms: Iterable[BaseGeometry],
) -> tuple[float, float, float, float] | None:
    """返回一组几何对象的边界范围。全部为空对象时返回 None。"""
    bounds = [geom.bounds for geom in geoms if not geom.is_empty]
    if bounds:
        bounds = np.array(bounds)
        x0 = bounds[:, 0].min()
        x1 = bounds[:, 2].max()
        y0 = bounds[:, 1].min()
        y1 = bounds[:, 3].max()
        return x0, x1, y0, y1
    else:
        return None


# TODO: 类持有对几何对象的引用，会存在问题吗？
class GeomCollection(PathCollection):
    """投影并绘制多边形对象的 Collection 类"""

    def __init__(
        self,
        ax: Axes,
        geoms: Collection[BaseGeometry],
        crs: ccrs.CRS | None,
        fast_transform: bool = True,
        skip_outside: bool = True,
        **kwargs: Any,
    ) -> None:
        self.geoms = geoms
        self.fast_transform = fast_transform
        self.skip_outside = skip_outside

        self.crs = crs
        self._on_geoaxes = isinstance(ax, GeoAxes)
        if self._on_geoaxes:
            if self.crs is None:
                self.crs = PLATE_CARREE
            self._geoms_to_paths = partial(
                _transform_geoms_to_paths,
                crs_from=self.crs,
                crs_to=ax.projection,
                fast_transform=fast_transform,
            )
        else:
            if self.crs is not None:
                raise ValueError("ax 不是 GeoAxes 时 crs 只能为 None")
            self._geoms_to_paths = _geoms_to_paths

        # 尝试用椭圆作为初始化的 paths
        paths = []
        if ax.get_autoscale_on():
            extents = _get_geoms_extents(self.geoms)
            if extents is not None:
                if self._on_geoaxes:
                    x0, x1, y0, y1 = extents
                    x = (x0 + x1) / 2
                    y = (y0 + y1) / 2
                    a = (x1 - x0) / 2
                    b = (y1 - y0) / 2
                    verts = make_ellipse(x, y, a, b, ccw=False)
                    ellipse = sgeom.Polygon(verts)
                    paths = self._geoms_to_paths([ellipse])
                else:
                    extent_path = fshp.box_path(*extents)
                    paths = [extent_path]

        super().__init__(paths, **kwargs)
        ax.add_collection(self)
        ax._request_autoscale_view()

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        if not self.get_visible():
            return None

        if self.skip_outside:
            # x0 == x1 == y0 == y1 也 OK
            if self._on_geoaxes:
                x0, x1, y0, y1 = self.axes.get_extent(self.crs)
            else:
                x0, x1 = sorted(self.axes.get_xlim())
                y0, y1 = sorted(self.axes.get_ylim())
            extent_polygon = sgeom.box(x0, y0, x1, y1)

            # 1 对应与边框有交点的多边形，需要做投影
            # 2 对应无交点的多边形，直接用占位符 Path 替代
            geom_dict1 = {}
            geom_dict2 = {}
            for i, geom in enumerate(self.geoms):
                if extent_polygon.intersects(geom):
                    geom_dict1[i] = geom
                else:
                    geom_dict2[i] = geom

            # 利用字典的有序性
            paths1 = self._geoms_to_paths(geom_dict1.values())
            paths2 = [fshp.EMPTY_PATH] * len(geom_dict2)
            path_dict1 = dict(zip(geom_dict1.keys(), paths1))
            path_dict2 = dict(zip(geom_dict2.keys(), paths2))
            path_dict = path_dict1 | path_dict2
            paths = [path for _, path in sorted(path_dict.items())]
        else:
            paths = self._geoms_to_paths(self.geoms)

        self.set_paths(paths)
        super().draw(renderer)


class TextCollection(Artist):
    """Text 集合"""

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        s: ArrayLike,
        skip_outside: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.skip_outside = skip_outside
        self.set_zorder(kwargs.get("zorder", 3))
        if "transform" in kwargs:
            self.set_transform(kwargs["transform"])

        self.x, self.y, self.s = map(np.atleast_1d, [x, y, s])
        if not len(self.x) == len(self.y) == len(self.s):
            raise ValueError("x、y 和 s 长度必须相同")
        self.coords = np.c_[self.x, self.y]

        kwargs = normalize_kwargs(kwargs, Text)
        kwargs.setdefault("horizontalalignment", "center")
        kwargs.setdefault("verticalalignment", "center")
        kwargs.setdefault("clip_on", True)
        self.texts = [
            Text(xi, yi, si, **kwargs) for xi, yi, si in zip(self.x, self.y, self.s)
        ]

    def set_figure(self, fig: Figure) -> None:
        super().set_figure(fig)
        for text in self.texts:
            text.set_figure(fig)

    def _set_clip_polygon(self, polygon: fshp.PolygonType) -> None:
        # 要求 polygon 基于 data 坐标系
        trans = self.get_transform() - self.axes.transData
        coords = trans.transform(self.coords)
        # contains 不包含落在边界上的点
        mask = contains(polygon, coords[:, 0], coords[:, 1])
        for i in np.nonzero(~mask)[0]:
            self.texts[i].set_visible(False)

    def _init(self) -> None:
        if not self.skip_outside:
            return None

        # 不画出坐标在边框外的 Text
        x0, x1 = sorted(self.axes.get_xlim())
        y0, y1 = sorted(self.axes.get_ylim())
        extent_polygon = sgeom.box(x0, y0, x1, y1)
        self._set_clip_polygon(extent_polygon)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        if not self.get_visible():
            return None

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
        qk_kwargs: dict | None = None,
        patch_kwargs: dict | None = None,
    ) -> None:
        self.units = units
        self.width = width
        self.height = height
        self.loc = loc

        if loc == "lower left":
            X = width / 2
            Y = height / 2
        elif loc == "lower right":
            X = 1 - width / 2
            Y = height / 2
        elif loc == "upper left":
            X = width / 2
            Y = 1 - height / 2
        elif loc == "upper right":
            X = 1 - width / 2
            Y = 1 - height / 2
        else:
            raise ValueError(
                "loc: {'lower left', 'lower right', 'upper left', 'upper right'}"
            )

        qk_kwargs = normalize_kwargs(qk_kwargs)
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

        # 将 qk 调整至 patch 的中心
        fontsize = self.text.get_fontsize() / 72
        dy = (self._labelsep_inches + fontsize) / 2
        trans = mtransforms.offset_copy(Q.axes.transAxes, Q.figure.figure, 0, dy)
        self.set_transform(trans)

        patch_kwargs = normalize_kwargs(patch_kwargs, Rectangle)
        patch_kwargs.setdefault("linewidth", 0.8)
        patch_kwargs.setdefault("edgecolor", "k")
        patch_kwargs.setdefault("facecolor", "w")
        self.patch = Rectangle(
            xy=(X - width / 2, Y - height / 2),
            width=width,
            height=height,
            transform=Q.axes.transAxes,
            **patch_kwargs,
        )

    def _set_transform(self) -> None:
        # 无效化 QuiveKey 的同名方法
        pass

    def set_figure(self, fig: Figure) -> None:
        self.patch.set_figure(fig)
        super().set_figure(fig)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
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
        pc_kwargs: dict | None = None,
        text_kwargs: dict | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.angle = angle
        self.size = size
        self.style = style

        head = size / 72
        if style == "arrow":
            width = axis = head * 2 / 3
            paths = self._make_paths(width, head, axis)
            colors = ["k", "w"]
        elif style == "circle":
            width = axis = head * 2 / 3
            radius = head * 2 / 5
            paths = [
                Path.circle((0, head / 9), radius),
                *self._make_paths(width, head, axis),
            ]
            colors = ["none", "k", "w"]
        elif style == "star":
            width = head / 3
            axis = head + width / 2
            paths = []
            path1, path2 = self._make_paths(width, head, axis)
            for deg in range(0, 360, 90):
                rotation = mtransforms.Affine2D().rotate_deg(deg)
                paths.append(path1.transformed(rotation))
                paths.append(path2.transformed(rotation))
            colors = ["k", "w"]
        else:
            raise ValueError("style: {'arrow', 'circle', 'star'}")

        pc_kwargs = normalize_kwargs(pc_kwargs, PathCollection)
        pc_kwargs.setdefault("linewidth", 1)
        pc_kwargs.setdefault("edgecolor", "k")
        pc_kwargs.setdefault("facecolor", colors)
        pc_kwargs.setdefault("clip_on", False)
        pc_kwargs.setdefault("zorder", 5)
        super().__init__(paths, transform=None, **pc_kwargs)

        # 文字在箭头上方
        pad = head / 2.5
        text_kwargs = normalize_kwargs(text_kwargs, Text)
        text_kwargs.setdefault("fontsize", size / 1.5)
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
    def _make_paths(width: float, head: float, axis: float) -> tuple[Path, Path]:
        """width：箭头宽度，head：箭头长度，axis：箭头中轴长度。且箭头方向朝上。"""
        path1 = Path([(0, 0), (0, axis), (-width / 2, axis - head), (0, 0)])
        path2 = Path([(0, 0), (width / 2, axis - head), (0, axis), (0, 0)])

        return path1, path2

    def _init(self) -> None:
        # 计算指北针的方向
        if self.angle is None:
            if isinstance(self.axes, GeoAxes):
                crs = PLATE_CARREE
                axes_to_data = self.axes.transAxes - self.axes.transData
                x0, y0 = axes_to_data.transform((self.x, self.y))
                crs.transform_point(x0, y0, self.axes.projection)
                lon0, lat0 = crs.transform_point(x0, y0, self.axes.projection)
                lon1, lat1 = lon0, min(lat0 + 0.01, 90)
                x1, y1 = self.axes.projection.transform_point(lon1, lat1, crs)
                theta = math.degrees(math.atan2(y1 - y0, x1 - x0))
                azimuth = t_to_az(theta, degrees=True)
            else:
                azimuth = 0
        else:
            azimuth = self.angle

        rotation = mtransforms.Affine2D().rotate_deg(-azimuth)
        translation = mtransforms.ScaledTranslation(self.x, self.y, self.axes.transAxes)
        trans = self.axes.figure.dpi_scale_trans + rotation + translation
        self.text.set_transform(trans)
        self.text.set_rotation(-azimuth)
        self.set_transform(trans)

    def set_figure(self, fig: Figure) -> None:
        self.text.set_figure(fig)
        super().set_figure(fig)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
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
        self.x = x
        self.y = y
        self.length = length
        self.units = units

        if units == "m":
            self._units = 1
        elif units == "km":
            self._units = 1000
        else:
            raise ValueError("units: {'m', 'km'}")

        super().__init__(ax.figure, (0, 0, 1, 1), zorder=5)
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
        # 在 data 坐标系取一条直线，计算单位长度对应的地理长度。
        if isinstance(self.axes, GeoAxes):
            # GeoAxes 取地图中心一段横线
            crs = PLATE_CARREE
            geod = crs.get_geod()
            xmin, xmax = self.axes.get_xlim()
            ymin, ymax = self.axes.get_ylim()
            xmid = (xmin + xmax) / 2
            ymid = (ymin + ymax) / 2
            dx = (xmax - xmin) / 10
            x0 = xmid - dx / 2
            x1 = xmid + dx / 2
            lon0, lat0 = crs.transform_point(x0, ymid, self.axes.projection)
            lon1, lat1 = crs.transform_point(x1, ymid, self.axes.projection)
            dr = geod.inv(lon0, lat0, lon1, lat1)[2] / self._units
            dxdr = dx / dr
        else:
            # Axes 认为是 PlateCarree 投影，取中心纬度。
            Re = 6371e3
            L = 2 * np.pi * Re / 360
            lat0, lat1 = self.axes.get_ylim()
            lat = (lat0 + lat1) / 2
            drdx = L * math.cos(math.radians(lat))
            dxdr = self._units / drdx

        # 重新设置比例尺的大小和位置
        axes_to_data = self.axes.transAxes - self.axes.transData
        data_to_figure = self.axes.transData - self.figure.transSubfigure
        x, y = axes_to_data.transform((self.x, self.y))
        width = self.length * dxdr
        bbox = mtransforms.Bbox.from_bounds(x, y, width, 1e-4 * width)
        bbox = data_to_figure.transform_bbox(bbox)
        self.set_position(bbox)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        self._init()
        super().draw(renderer)


# TODO: 非矩形边框
class Frame(Artist):
    """GMT风格的边框"""

    def __init__(self, width: float = 5, **kwargs: Any) -> None:
        self.width = width
        self._width = width / 72
        super().__init__()
        self.set_zorder(2.5)

        kwargs = normalize_kwargs(kwargs, PathCollection)
        kwargs.setdefault("linewidth", 1)
        kwargs.setdefault("edgecolor", "k")
        kwargs.setdefault("facecolor", ["k", "w"])

        # 暂时用空 paths 占位
        self.pcs = {}
        for key in ["left", "right", "top", "bottom", "corner"]:
            self.pcs[key] = PathCollection([], transform=None, **kwargs)

    def _init(self) -> None:
        if isinstance(self.axes, GeoAxes) and not isinstance(
            self.axes.projection, (ccrs.PlateCarree, ccrs.Mercator)
        ):
            raise ValueError("只支持 PlateCarree 和 Mercator 投影")

        # 将 inches 坐标系的 width 转为 axes 坐标系里的 dx 和 dy
        inches_to_axes = self.axes.figure.dpi_scale_trans - self.axes.transAxes
        mtx = inches_to_axes.get_matrix()
        dx = self._width * mtx[0, 0]
        dy = self._width * mtx[1, 1]

        xtrans = self.axes.get_xaxis_transform()
        ytrans = self.axes.get_yaxis_transform()
        self.pcs["top"].set_transform(xtrans)
        self.pcs["bottom"].set_transform(xtrans)
        self.pcs["left"].set_transform(ytrans)
        self.pcs["right"].set_transform(ytrans)
        self.pcs["corner"].set_transform(self.axes.transAxes)

        # 收集 xlim 和 ylim 范围内所有刻度并去重
        major_xticks = self.axes.xaxis.get_majorticklocs()
        minor_xticks = self.axes.xaxis.get_minorticklocs()
        major_yticks = self.axes.yaxis.get_majorticklocs()
        minor_yticks = self.axes.yaxis.get_minorticklocs()
        xmin, xmax = sorted(self.axes.get_xlim())
        ymin, ymax = sorted(self.axes.get_ylim())
        xticks = np.array([xmin, xmax, *major_xticks, *minor_xticks])
        yticks = np.array([ymin, ymax, *major_yticks, *minor_yticks])
        xticks = xticks[(xticks >= xmin) & (xticks <= xmax)]
        yticks = yticks[(yticks >= ymin) & (yticks <= ymax)]
        # 通过 round 抵消 central_longitude 导致的浮点误差
        xticks = np.unique(xticks.round(9))
        yticks = np.unique(yticks.round(9))
        nx = len(xticks)
        ny = len(yticks)

        top_paths = [
            fshp.box_path(xticks[i], xticks[i + 1], 1, 1 + dy) for i in range(nx - 1)
        ]
        # 即便 xaxis 方向反转也维持边框的颜色顺序
        if self.axes.xaxis.get_inverted():
            top_paths.reverse()
        self.pcs["top"].set_paths(top_paths)

        # 比例尺对象只设置上边框
        if isinstance(self.axes, ScaleBar):
            return None

        bottom_paths = [
            fshp.box_path(xticks[i], xticks[i + 1], -dy, 0) for i in range(nx - 1)
        ]
        if self.axes.xaxis.get_inverted():
            bottom_paths.reverse()
        self.pcs["bottom"].set_paths(bottom_paths)

        left_paths = [
            fshp.box_path(-dx, 0, yticks[i], yticks[i + 1]) for i in range(ny - 1)
        ]
        if self.axes.yaxis.get_inverted():
            left_paths.reverse()
        self.pcs["left"].set_paths(left_paths)

        right_paths = [
            fshp.box_path(1, 1 + dx, yticks[i], yticks[i + 1]) for i in range(ny - 1)
        ]
        if self.axes.yaxis.get_inverted():
            right_paths.reverse()
        self.pcs["right"].set_paths(right_paths)

        # 单独画出四个角落的方块
        corner_paths = [
            fshp.box_path(-dx, 0, -dy, 0),
            fshp.box_path(1, 1 + dx, -dy, 0),
            fshp.box_path(-dx, 0, 1, 1 + dy),
            fshp.box_path(1, 1 + dx, 1, 1 + dy),
        ]
        self.pcs["corner"].set_paths(corner_paths)
        fc = self.pcs["top"].get_facecolor()[-1]
        self.pcs["corner"].set_facecolor(fc)

    def set_figure(self, fig: Figure) -> None:
        super().set_figure(fig)
        for pc in self.pcs.values():
            pc.set_figure(fig)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        if not self.get_visible():
            return None

        self._init()
        for pc in self.pcs.values():
            pc.draw(renderer)
        self.stale = False


class CurlyQuiver:
    def __init__(self) -> None:
        raise NotImplementedError
