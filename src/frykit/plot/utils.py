from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
import shapely
from matplotlib.axes._base import _AxesBase
from matplotlib.path import Path
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from frykit.calc import is_finite
from frykit.shp.typing import GeometryT, PolygonType
from frykit.utils import format_type_error

__all__ = [
    "box_path",
    "EMPTY_PATH",
    "EMPTY_POLYGON",
    "geometry_to_path",
    "path_to_polygon",
    "make_transformer",
    "project_geometry",
    "get_axes_extents",
]


def box_path(x0: float, x1: float, y0: float, y1: float, ccw: bool = True) -> Path:
    """构造方框 Path"""
    vertices = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    if ccw:
        vertices.reverse()
    return Path(vertices, closed=True)


EMPTY_PATH = Path(np.empty((0, 2)))
EMPTY_POLYGON = shapely.Polygon()


def _line_string_codes(line_string: shapely.LineString) -> list[np.uint8]:
    n = shapely.get_num_points(line_string)
    return [Path.MOVETO, *([Path.LINETO] * (n - 1))]


def _linear_ring_codes(linear_ring: shapely.LinearRing) -> list[np.uint8]:
    n = shapely.get_num_points(linear_ring)
    return [Path.MOVETO, *([Path.LINETO] * (n - 2)), Path.CLOSEPOLY]


def geometry_to_path(geometry: BaseGeometry) -> Path:
    """
    将几何对象转为 Matplotlib 的 Path 对象

    - 空多边形对应空 Path
    - Polygon 会先调整为外环顺时针，内环逆时针，再转为 Path。
    - 多成员的几何对象会用 Path.make_compound_path 合并

    See Also
    --------
    cartopy.mpl.patch.geos_to_path
    """
    if isinstance(geometry, BaseGeometry) and geometry.is_empty:
        return EMPTY_PATH

    match geometry:
        case shapely.Point():
            return Path([[geometry.x, geometry.y]], [Path.MOVETO])

        case shapely.LinearRing():
            return Path(shapely.get_coordinates(geometry), _linear_ring_codes(geometry))

        case shapely.LineString():
            return Path(shapely.get_coordinates(geometry), _line_string_codes(geometry))

        # 直接使用 orient 效率太低
        case shapely.Polygon():
            vertices, codes = [], []
            for i, linear_ring in enumerate([geometry.exterior, *geometry.interiors]):
                coords = shapely.get_coordinates(linear_ring)
                if (i == 0) == linear_ring.is_ccw:
                    coords = coords[::-1]
                vertices.append(coords)
                codes.extend(_linear_ring_codes(linear_ring))

            vertices = np.vstack(vertices)
            return Path(vertices, codes)

        case BaseMultipartGeometry():
            return Path.make_compound_path(*map(geometry_to_path, geometry.geoms))

        case _:
            raise TypeError(format_type_error("geometry", geometry, BaseGeometry))


def path_to_polygon(path: Path) -> PolygonType:
    """
    将 Matplotlib 的 Path 对象转为多边形对象

    - 空 Path 对应空多边形
    - 若坐标含 nan 或 inf，那么整个多边形会变成空多边形。
    - 要求输入是 geometry_to_path(polygon) 的结果，其它输入不保证结果正确。

    See Also
    --------
    cartopy.mpl.patch.path_to_geos
    """
    if len(path.vertices) == 0:  # type: ignore
        return EMPTY_POLYGON

    invalid_flag = False
    collection: list[tuple[shapely.LinearRing, list[shapely.LinearRing]]] = []
    indices = np.nonzero(path.codes == Path.MOVETO)[0][1:]
    for vertices in np.vsplit(path.vertices, indices):
        if not is_finite(vertices):
            invalid_flag = True
            continue
        linear_ring = shapely.LinearRing(vertices)
        if linear_ring.is_ccw:
            if not invalid_flag:
                assert len(collection) > 0
                collection[-1][1].append(linear_ring)
        else:
            collection.append((linear_ring, []))
            invalid_flag = False

    polygons = [shapely.Polygon(shell, holes) for shell, holes in collection]
    match len(polygons):
        case 0:
            return EMPTY_POLYGON
        case 1:
            return polygons[0]
        case _:
            return shapely.MultiPolygon(polygons)


def _transform_geometry(
    geometry: GeometryT, transform: Callable[[NDArray], NDArray]
) -> GeometryT:
    """shapely.ops.transform 的修改版，会将变换后坐标含 nan 或 inf 的几何对象设为空对象。"""
    if isinstance(geometry, BaseGeometry) and geometry.is_empty:
        return geometry

    match geometry:
        # Point 可以接受形如 (1, 2) 的坐标
        case shapely.Point() | shapely.LineString() | shapely.LinearRing():
            coords = transform(shapely.get_coordinates(geometry))
            if is_finite(coords):
                return type(geometry)(coords)
            else:
                return type(geometry)()

        case shapely.Polygon():
            shell = transform(shapely.get_coordinates(geometry.exterior))
            if not is_finite(shell):
                return type(geometry)()

            holes = []
            for interior in geometry.interiors:
                hole = transform(shapely.get_coordinates(interior))
                if is_finite(hole):
                    holes.append(hole)

            return type(geometry)(shell, holes)

        case BaseMultipartGeometry():
            parts = [
                _transform_geometry(part, transform)
                for part in geometry.geoms
                if not part.is_empty
            ]
            return type(geometry)(parts)  # type: ignore

        case _:
            raise TypeError(format_type_error("geometry", geometry, BaseGeometry))


@lru_cache
def make_transformer(crs_from: CRS, crs_to: CRS) -> Transformer:
    """创建 pyproj 的 Transformer 对象"""
    return Transformer.from_crs(crs_from, crs_to, always_xy=True)


def project_geometry(geometry: GeometryT, crs_from: CRS, crs_to: CRS) -> GeometryT:
    """对几何对象做投影"""
    if crs_from == crs_to:
        return geometry  # 直接返回不可变对象

    transformer = make_transformer(crs_from, crs_to)

    def transform_coords(coords: NDArray) -> NDArray:
        return np.column_stack(transformer.transform(coords[:, 0], coords[:, 1]))

    return _transform_geometry(geometry, transform_coords)


def get_axes_extents(ax: _AxesBase) -> tuple[float, float, float, float]:
    """获取 Axes 在 data 坐标系的 extents"""
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    return x0, x1, y0, y1
