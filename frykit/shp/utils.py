from __future__ import annotations

from functools import partial
from itertools import chain
from typing import Callable, TypeVar, cast, overload

import numpy as np
import pandas as pd
import shapefile
import shapely
import shapely.geometry as sgeom
from matplotlib.path import Path
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from frykit.calc import is_finite
from frykit.shp.typing import (
    FeatureDict,
    GeoJSONDict,
    GeometryCollectionDict,
    GeometryDict,
    LineStringDict,
    MultiLineStringCoordinates,
    MultiLineStringDict,
    MultiPointCoordinates,
    MultiPointDict,
    MultiPolygonDict,
    PointCoordinates,
    PointDict,
    PolygonCoordinates,
    PolygonDict,
    PolygonType,
)
from frykit.typing import PathType
from frykit.utils import format_type_error


@overload
def orient_polygon(polygon: shapely.Polygon, ccw: bool = True) -> shapely.Polygon: ...


@overload
def orient_polygon(
    polygon: shapely.MultiPolygon, ccw: bool = True
) -> shapely.MultiPolygon: ...


def orient_polygon(polygon: PolygonType, ccw: bool = True) -> PolygonType:
    """调整多边形的绕行方向。例如 ccw=True 时外环逆时针，内环顺时针。"""
    sign = 1 if ccw else -1
    if isinstance(polygon, shapely.Polygon):
        return orient(polygon, sign)
    elif isinstance(polygon, shapely.MultiPolygon):
        return shapely.MultiPolygon([orient(part, sign) for part in polygon.geoms])
    else:
        raise TypeError(
            format_type_error(
                "polygon", polygon, [shapely.Polygon, shapely.MultiPolygon]
            )
        )


@overload
def geometry_to_shape(geometry: shapely.Point) -> PointCoordinates: ...


@overload
def geometry_to_shape(geometry: shapely.MultiPoint) -> MultiPointCoordinates: ...


@overload
def geometry_to_shape(
    geometry: shapely.LineString | shapely.MultiLineString,
) -> MultiLineStringCoordinates: ...


@overload
def geometry_to_shape(geometry: PolygonType) -> PolygonCoordinates: ...


def geometry_to_shape(
    geometry: BaseGeometry,
) -> (
    PointCoordinates
    | MultiPointCoordinates
    | MultiLineStringCoordinates
    | PolygonCoordinates
):
    """几何对象转为 shapeifle 适用的坐标序列"""

    def get_coordinates(geometry: BaseGeometry) -> list[list[float]]:
        return shapely.get_coordinates(geometry).tolist()

    def polygon_to_shape(polygon: shapely.Polygon) -> PolygonCoordinates:
        polygon = orient(polygon, sign=-1)
        polys = [get_coordinates(polygon.exterior)]
        polys.extend(list(map(get_coordinates, polygon.interiors)))

        return polys

    match geometry:
        case shapely.Point():
            return [geometry.x, geometry.y]
        case shapely.MultiPoint():
            return get_coordinates(geometry)
        case shapely.LinearRing():
            raise TypeError(
                "geometry 是 shapely.LinearRing 类型，"
                "需要先转换成 shapely.LineString 或 shapely.Polygon 类型"
            )
        case shapely.LineString():
            return [get_coordinates(geometry)]
        case shapely.MultiLineString():
            return list(map(get_coordinates, geometry.geoms))
        case shapely.Polygon():
            return polygon_to_shape(geometry)
        case shapely.MultiPolygon():
            return list(chain(*map(polygon_to_shape, geometry.geoms)))
        case shapely.GeometryCollection():
            raise TypeError(
                "geometry 是 shapely.GeometryCollection 类型，在 shapefile 中没有直接对应的类型"
            )
        case _:
            raise TypeError(format_type_error("geometry", geometry, BaseGeometry))


@overload
def geometry_to_dict(geometry: shapely.Point) -> PointDict: ...


@overload
def geometry_to_dict(geometry: shapely.MultiPoint) -> MultiPointDict: ...


@overload
def geometry_to_dict(geometry: shapely.LineString) -> LineStringDict: ...


@overload
def geometry_to_dict(geometry: shapely.MultiLineString) -> MultiLineStringDict: ...


@overload
def geometry_to_dict(geometry: shapely.Polygon) -> PolygonDict: ...


@overload
def geometry_to_dict(geometry: shapely.MultiPolygon) -> MultiPolygonDict: ...


@overload
def geometry_to_dict(
    geometry: shapely.GeometryCollection,
) -> GeometryCollectionDict: ...


def geometry_to_dict(geometry: BaseGeometry) -> GeometryDict:
    """几何对象转为 GeoJSON 的 geometry 字典"""
    if not isinstance(geometry, BaseGeometry):
        raise TypeError(format_type_error("geometry", geometry, BaseGeometry))

    if isinstance(geometry, shapely.LinearRing):
        raise TypeError(
            "geometry 是 shapely.LinearRing 类型，"
            "需要先转换成 shapely.LineString 或 shapely.Polygon 类型"
        )

    if isinstance(geometry, (shapely.Polygon, shapely.MultiPolygon)):
        geometry = orient_polygon(geometry)
    geometry_dict = sgeom.mapping(geometry)

    return cast(GeometryDict, geometry_dict)


def get_geojson_properties(geojson_dict: GeoJSONDict) -> pd.DataFrame:
    """提取 GeoJSON 字典里的所有 properties 为 DataFrame"""
    records = [feature["properties"] for feature in geojson_dict["features"]]
    return pd.DataFrame.from_records(records)


def get_geojson_geometries(geojson_dict: GeoJSONDict) -> list[BaseGeometry]:
    """提取 GeoJSON 字典里的所有几何对象"""
    return [sgeom.shape(feature["geometry"]) for feature in geojson_dict["features"]]  # type: ignore


def get_shapefile_properties(filepath: PathType) -> pd.DataFrame:
    """提取 shapefile 文件里的所有属性为 DataFrame"""
    with shapefile.Reader(filepath) as reader:
        records = [record.as_dict() for record in reader.iterRecords()]
    return pd.DataFrame.from_records(records)


def get_shapefile_geometries(filepath: PathType) -> list[BaseGeometry]:
    """提取 shapefile 文件里的所有几何对象"""
    with shapefile.Reader(filepath) as reader:
        return list(map(sgeom.shape, reader.iterShapes()))


def get_representative_xy(geometry: BaseGeometry) -> tuple[float, float]:
    """计算保证在几何对象内部的代表点"""
    point = geometry.representative_point()
    return point.x, point.y


def make_feature(geometry_dict: GeometryDict, properties: dict) -> FeatureDict:
    """用 geometry 和 properties 字典构造 GeoJSON 的 feature 字典"""
    return {"type": "Feature", "geometry": geometry_dict, "properties": properties}


def make_geojson(features: list[FeatureDict]) -> GeoJSONDict:
    """用一组 feature 字典构造 GeoJSON 字典"""
    return {"type": "FeatureCollection", "features": features}


# 用于占位的 Path，不会被画出。
EMPTY_PATH = Path(np.empty((0, 2)))
EMPTY_POLYGON = shapely.Polygon()


def _line_string_codes(line_string: shapely.LineString) -> list[np.uint8]:
    n = shapely.get_num_points(line_string)
    return [Path.MOVETO, *([Path.LINETO] * (n - 1))]


def _linear_ring_codes(linear_ring: shapely.LinearRing) -> list[np.uint8]:
    n = shapely.get_num_points(linear_ring)
    return [Path.MOVETO, *([Path.LINETO] * (n - 2)), Path.CLOSEPOLY]


def geom_to_path(geom: BaseGeometry) -> Path:
    """
    几何对象转为 Path

    - 空几何对象对应空 Path
    - Point 和 LineString 保留原来的坐标顺序
    - LinearRing 对应的 Path 里坐标沿顺时针绕行
    - Polygon 对应的 Path 里外环坐标沿顺时针绕行，内环坐标沿逆时针绕行。
    - Multi 对象和 GeometryCollection 使用 Path.make_compound_path

    See Also
    --------
    cartopy.mpl.patch.geos_to_path
    """
    match geom:
        case shapely.Point():
            if not geom.is_empty:
                return Path([[geom.x, geom.y]], [Path.MOVETO])
            else:
                return EMPTY_PATH

        case shapely.LineString():
            if not geom.is_empty:
                return Path(shapely.get_coordinates(geom), _line_string_codes(geom))
            else:
                return EMPTY_PATH

        # orient_polygon 效率偏低
        case shapely.Polygon():
            if geom.is_empty:
                return EMPTY_PATH

            vertices = []
            codes = []
            exterior = geom.exterior
            coords = shapely.get_coordinates(exterior)
            if exterior.is_ccw:
                coords = coords[::-1]
            vertices.append(coords)
            codes.extend(_linear_ring_codes(exterior))

            for interior in geom.interiors:
                coords = shapely.get_coordinates(interior)
                if not interior.is_ccw:
                    coords = coords[::-1]
                vertices.append(coords)
                codes.extend(_linear_ring_codes(interior))

            vertices = np.vstack(vertices)
            return Path(vertices, codes)

        case (
            shapely.MultiPoint()
            | shapely.MultiLineString()
            | shapely.MultiPolygon()
            | shapely.GeometryCollection()
        ):
            if not geom.is_empty:
                return Path.make_compound_path(*map(geom_to_path, geom.geoms))
            else:
                return EMPTY_PATH

        case _:
            raise TypeError(format_type_error("geom", geom, BaseGeometry))


def path_to_polygon(path: Path) -> PolygonType:
    """
    Path 转为多边形对象

    - 空 Path 对应空多边形
    - 要求输入是 geom_to_path(polygon) 的结果，其它输入不保证结果正确。
    - 含 nan 或 inf 的部分对应空多边形

    See Also
    --------
    cartopy.mpl.patch.path_to_geos
    """
    if len(path.vertices) == 0:  # type: ignore
        return EMPTY_POLYGON

    collection = []
    invalid_flag = False
    indices = np.nonzero(path.codes == Path.MOVETO)[0][1:]
    for vertices in np.vsplit(path.vertices, indices):
        if not is_finite(vertices):
            invalid_flag = True
            continue
        linear_ring = shapely.LinearRing(vertices)
        if linear_ring.is_ccw:
            if not invalid_flag:
                collection[-1][1].append(linear_ring)
        else:
            collection.append((linear_ring, []))
            invalid_flag = False

    polygons = [shapely.Polygon(shell, holes) for shell, holes in collection]
    if len(polygons) == 0:
        return EMPTY_POLYGON
    elif len(polygons) == 1:
        return polygons[0]
    else:
        return shapely.MultiPolygon(polygons)


def box_path(x0: float, x1: float, y0: float, y1: float) -> Path:
    """构造顺时针绕行的方框 Path"""
    vertices = [(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)]
    return Path(vertices, closed=True)


GeometryT = TypeVar("GeometryT", bound=BaseGeometry)


def _transform(func: Callable[[NDArray], NDArray], geom: GeometryT) -> GeometryT:
    """shapely.ops.transform 的修改版，会将坐标含无效值的对象设为空对象。"""
    match geom:
        case shapely.Point() | shapely.LineString() | shapely.LinearRing():
            geom_type = type(geom)
            if geom.is_empty:
                return geom_type()

            coords = func(shapely.get_coordinates(geom))
            if is_finite(coords):
                return geom_type(coords)
            else:
                return geom_type()

        case shapely.Polygon():
            geom_type = type(geom)
            if geom.is_empty:
                return geom_type()

            shell = func(shapely.get_coordinates(geom.exterior))
            if not is_finite(shell):
                return geom_type()

            holes = []
            for interior in geom.interiors:
                hole = func(shapely.get_coordinates(interior))
                if is_finite(hole):
                    holes.append(hole)

            return geom_type(shell, holes)

        case (
            shapely.MultiPoint()
            | shapely.MultiLineString()
            | shapely.MultiPolygon()
            | shapely.GeometryCollection()
        ):
            geom_type = type(geom)
            if geom.is_empty:
                return geom_type()

            parts = []
            for part in geom.geoms:
                part = _transform(func, part)
                if not part.is_empty:
                    parts.append(part)

            if len(parts) > 0:
                return geom_type(parts)
            else:
                return geom_type()

        case _:
            raise TypeError(format_type_error("geom", geom, BaseGeometry))


class GeometryTransformer:
    """
    对几何对象做坐标变换的类

    基于 pyproj.Transformer 实现，可能在地图边界出现错误的结果。
    如果变换后的坐标存在 nan 或 inf，会将对应的几何对象设为空对象。

    See Also
    --------
    cartopy.crs.Projection.project_geometry
    """

    def __init__(self, crs_from: CRS, crs_to: CRS) -> None:
        """
        Parameters
        ----------
        crs_from : CRS
            源坐标系

        crs_to : CRS
            目标坐标系
        """
        self.crs_from = crs_from
        self.crs_to = crs_to

        # 坐标系相同时直接复制
        if crs_from == crs_to:
            self._func = lambda x: type(x)(x)
            return None

        # 避免反复创建 Transformer 的开销
        transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)

        def func(coords: NDArray) -> NDArray:
            return np.column_stack(
                transformer.transform(coords[:, 0], coords[:, 1])
            ).squeeze()

        self._func = partial(_transform, func)

    def __call__(self, geom: GeometryT) -> GeometryT:
        """
        对几何对象做变换

        Parameters
        ----------
        geom : BaseGeometry
            源坐标系上的几何对象

        Returns
        -------
        geom : BaseGeometry
            目标坐标系上的几何对象
        """
        return self._func(geom)
