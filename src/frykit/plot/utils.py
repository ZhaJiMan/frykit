from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Sequence, overload

import numpy as np
import shapely
from matplotlib.axes._base import _AxesBase
from matplotlib.path import Path
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from frykit.shp.typing import GeometryT, PolygonType
from frykit.utils import format_type_error

__all__ = [
    "EMPTY_PATH",
    "EMPTY_POLYGON",
    "box_path",
    "geometry_to_path",
    "get_axes_extents",
    "make_transformer",
    "path_to_polygon",
    "project_geometry",
]


def box_path(x0: float, x1: float, y0: float, y1: float, ccw: bool = True) -> Path:
    """构造方框 Path"""
    vertices = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    if ccw:
        vertices.reverse()
    return Path(vertices, closed=True)


EMPTY_PATH = Path(np.empty([0, 2]))
EMPTY_POLYGON = shapely.Polygon()


def _line_string_codes(n: int) -> NDArray[np.uint8]:
    codes = np.zeros(n, dtype=np.uint8)
    codes[0] = Path.MOVETO
    codes[1:] = Path.LINETO

    return codes


def _linear_ring_codes(n: int) -> NDArray[np.uint8]:
    codes = np.zeros(n, dtype=np.uint8)
    codes[0] = Path.MOVETO
    codes[1:-1] = Path.LINETO
    codes[-1] = Path.CLOSEPOLY

    return codes


def geometry_to_path(geometry: BaseGeometry) -> Path:
    """
    将几何对象转为 Matplotlib 的 Path 对象

    - 空多边形对应空 Path
    - Polygon 会先调整为外环顺时针，内环逆时针，再转为 Path。
    - 多成员的几何对象会用 Path.make_compound_path 合并

    See Also
    --------
    - cartopy.mpl.patch.geos_to_path
    - cartopy.mpl.path.shapely_to_path
    """
    if isinstance(geometry, BaseGeometry) and geometry.is_empty:
        return EMPTY_PATH

    match geometry:
        case shapely.Point():
            return Path([[geometry.x, geometry.y]], [Path.MOVETO])

        case shapely.LinearRing():
            coords = shapely.get_coordinates(geometry)
            return Path(coords, _linear_ring_codes(len(coords)))

        case shapely.LineString():
            coords = shapely.get_coordinates(geometry)
            return Path(coords, _line_string_codes(len(coords)))

        # 直接使用 orient 效率太低
        case shapely.Polygon():
            coords_list: list[NDArray[np.float64]] = []
            codes_list: list[NDArray[np.uint8]] = []
            for i, linear_ring in enumerate([geometry.exterior, *geometry.interiors]):
                coords = shapely.get_coordinates(linear_ring)
                if (i == 0) == linear_ring.is_ccw:
                    coords = coords[::-1]
                coords_list.append(coords)
                codes_list.append(_linear_ring_codes(len(coords)))

            vertices = np.vstack(coords_list)
            codes = np.concatenate(codes_list)
            return Path(vertices, codes)

        case BaseMultipartGeometry():
            return Path.make_compound_path(*map(geometry_to_path, geometry.geoms))

        case _:
            raise TypeError(format_type_error("geometry", geometry, BaseGeometry))


def path_to_polygon(path: Path) -> PolygonType:
    """
    将 Matplotlib 的 Path 对象转为多边形对象

    - 空 Path 对应空多边形
    - 要求输入是 geometry_to_path(polygon) 的结果，其它输入可能产生错误。

    See Also
    --------
    - cartopy.mpl.patch.path_to_geos
    - cartopy.mpl.path.path_to_shapely
    """
    if len(path.vertices) == 0:  # type: ignore
        return EMPTY_POLYGON

    collection: list[tuple[shapely.LinearRing, list[shapely.LinearRing]]] = []
    indices = np.nonzero(path.codes == Path.MOVETO)[0][1:]  # type: ignore
    for coords in np.vsplit(path.vertices, indices):
        linear_ring = shapely.LinearRing(coords)
        if linear_ring.is_ccw:
            collection[-1][1].append(linear_ring)
        else:
            collection.append((linear_ring, []))

    polygons = [shapely.Polygon(shell, holes) for shell, holes in collection]
    match len(polygons):
        case 0:
            return EMPTY_POLYGON
        case 1:
            return polygons[0]
        case _:
            return shapely.MultiPolygon(polygons)


@lru_cache
def make_transformer(crs_from: CRS, crs_to: CRS) -> Transformer:
    """创建 pyproj 的 Transformer 对象"""
    return Transformer.from_crs(crs_from, crs_to, always_xy=True)


def _make_transformation(
    crs_from: CRS, crs_to: CRS
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    # shapely>=2.1.0 且 interleaved=False 时就不需要这个辅助函数了
    def transformation(coords: NDArray[np.float64]) -> NDArray[np.float64]:
        transformer = make_transformer(crs_from, crs_to)
        return np.column_stack(transformer.transform(coords[:, 0], coords[:, 1]))

    return transformation


@overload
def project_geometry(geometry: GeometryT, crs_from: CRS, crs_to: CRS) -> GeometryT: ...


@overload
def project_geometry(
    geometry: Sequence[GeometryT], crs_from: CRS, crs_to: CRS
) -> NDArray[np.object_]: ...


@overload
def project_geometry(
    geometry: NDArray[np.object_], crs_from: CRS, crs_to: CRS
) -> NDArray[np.object_]: ...


def project_geometry(
    geometry: GeometryT | Sequence[GeometryT] | NDArray[np.object_],
    crs_from: CRS,
    crs_to: CRS,
) -> GeometryT | NDArray[np.object_]:
    """对一个或一组几何对象做投影。通过直接对坐标数组应用 pyproj 实现。"""
    geometries = np.array(geometry, dtype=np.object_)

    # 尽管 pyproj.Transformer 也能处理 crs 相等的情况，但还是有明显的性能损失
    if crs_from != crs_to:
        transformation = _make_transformation(crs_from, crs_to)
        geometries = shapely.transform(geometries, transformation)

    # 要模仿 shapely 对数组的处理吗？
    if geometries.ndim == 0 and not isinstance(geometry, np.ndarray):
        return geometries.item()
    else:
        return geometries


def get_axes_extents(ax: _AxesBase) -> tuple[float, float, float, float]:
    """获取 Axes 在 data 坐标系的 extents"""
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    return x0, x1, y0, y1
