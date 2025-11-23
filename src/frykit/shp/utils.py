from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import chain
from typing import Any, cast, overload

import shapely
import shapely.geometry as sgeom
from shapely.geometry.base import BaseGeometry

from frykit.shp.typing import (
    FeatureDict,
    GeoJSONDict,
    GeometryCollectionDict,
    GeometryDict,
    LineStringCoordinates,
    LineStringDict,
    LineStringType,
    MultiLineStringCoordinates,
    MultiLineStringDict,
    MultiPointCoordinates,
    MultiPointDict,
    MultiPolygonCoordinates,
    MultiPolygonDict,
    PointCoordinates,
    PointDict,
    PolygonCoordinates,
    PolygonDict,
    PolygonType,
)
from frykit.utils import format_type_error

__all__ = [
    "dict_to_geometry",
    "geometry_to_dict",
    "geometry_to_shape",
    "make_feature",
    "make_geojson",
    "orient_polygon",
]


def _orient(polygon: shapely.Polygon, ccw: bool = True) -> shapely.Polygon:
    if polygon.is_empty:
        return polygon

    # 循环比向量化略快？
    linear_rings: list[shapely.LinearRing] = []
    for i, linear_ring in enumerate([polygon.exterior, *polygon.interiors]):
        if ((i == 0) == ccw) != linear_ring.is_ccw:
            linear_ring = shapely.reverse(linear_ring)
        linear_rings.append(linear_ring)
    polygon = shapely.Polygon(linear_rings[0], linear_rings[1:])

    return polygon


@overload
def orient_polygon(polygon: shapely.Polygon, ccw: bool = True) -> shapely.Polygon: ...


@overload
def orient_polygon(
    polygon: shapely.MultiPolygon, ccw: bool = True
) -> shapely.MultiPolygon: ...


def orient_polygon(polygon: PolygonType, ccw: bool = True) -> PolygonType:
    """
    调整多边形的绕行方向

    ccw=True 时外环逆时针，内环顺时针；ccw=False 时外环顺时针，内环逆时针。

    See Also
    --------
    shapely.orient_polygons
    """
    if not isinstance(polygon, (shapely.Polygon, shapely.MultiPolygon)):
        raise TypeError(
            format_type_error(
                "polygon", polygon, [shapely.Polygon, shapely.MultiPolygon]
            )
        )

    if hasattr(shapely, "orient_polygons"):
        return cast(PolygonType, shapely.orient_polygons(polygon, exterior_cw=not ccw))

    if isinstance(polygon, shapely.Polygon):
        return _orient(polygon, ccw)
    else:
        return shapely.MultiPolygon([_orient(part, ccw) for part in polygon.geoms])


def _point_to_coordinates(point: shapely.Point) -> PointCoordinates:
    return [point.x, point.y]


def _multi_point_to_coordinates(
    multi_point: shapely.MultiPoint,
) -> MultiPointCoordinates:
    return shapely.get_coordinates(multi_point).tolist()


def _line_string_to_coordinates(
    line_string: shapely.LineString,
) -> LineStringCoordinates:
    return shapely.get_coordinates(line_string).tolist()


def _multi_line_string_to_coordinates(
    multi_line_string: shapely.MultiLineString,
) -> MultiLineStringCoordinates:
    return list(map(_line_string_to_coordinates, multi_line_string.geoms))


def _polygon_to_coordinates(
    polygon: shapely.Polygon, ccw: bool = True
) -> PolygonCoordinates:
    polygon = _orient(polygon, ccw)
    linear_rings = [polygon.exterior, *polygon.interiors]
    return list(map(_line_string_to_coordinates, linear_rings))


def _multi_polygon_to_coordinates(
    multi_polygon: shapely.MultiPolygon, ccw: bool = True
) -> MultiPolygonCoordinates:
    return [_polygon_to_coordinates(polygon, ccw) for polygon in multi_polygon.geoms]


@overload
def geometry_to_shape(geometry: shapely.Point) -> PointCoordinates: ...


@overload
def geometry_to_shape(geometry: shapely.MultiPoint) -> MultiPointCoordinates: ...


@overload
def geometry_to_shape(geometry: LineStringType) -> MultiLineStringCoordinates: ...


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
    match geometry:
        case shapely.Point():
            return _point_to_coordinates(geometry)
        case shapely.MultiPoint():
            return _multi_point_to_coordinates(geometry)
        case shapely.LinearRing():
            raise TypeError(
                "geometry 是 shapely.LinearRing 类型，"
                "需要先转换成 shapely.LineString 或 shapely.Polygon 类型"
            )
        case shapely.LineString():
            return [_line_string_to_coordinates(geometry)]
        case shapely.MultiLineString():
            return _multi_line_string_to_coordinates(geometry)
        case shapely.Polygon():
            return _polygon_to_coordinates(geometry, ccw=False)
        case shapely.MultiPolygon():
            return list(
                chain.from_iterable(_multi_polygon_to_coordinates(geometry, ccw=False))
            )
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
    """几何对象转为 GeoJSON 的 geometry 字典

    多边形会调整至外环逆时针，内环顺时针。

    See Also
    --------
    shapely.geometry.mapping
    """
    match geometry:
        case shapely.Point():
            coordinates = _point_to_coordinates(geometry)
        case shapely.MultiPoint():
            coordinates = _multi_point_to_coordinates(geometry)
        case shapely.LinearRing():
            raise TypeError(
                "geometry 是 shapely.LinearRing 类型，"
                "需要先转换成 shapely.LineString 或 shapely.Polygon 类型"
            )
        case shapely.LineString():
            coordinates = _line_string_to_coordinates(geometry)
        case shapely.MultiLineString():
            coordinates = _multi_line_string_to_coordinates(geometry)
        case shapely.Polygon():
            coordinates = _polygon_to_coordinates(geometry, ccw=True)
        case shapely.MultiPolygon():
            coordinates = _multi_polygon_to_coordinates(geometry, ccw=True)
        case shapely.GeometryCollection():
            return {
                "type": "GeometryCollection",
                "geometries": list(map(geometry_to_dict, geometry.geoms)),
            }
        case _:
            raise TypeError(format_type_error("geometry", geometry, BaseGeometry))

    return cast(GeometryDict, {"type": geometry.geom_type, "coordinates": coordinates})


@overload
def dict_to_geometry(geometry_dict: PointDict) -> shapely.Point: ...


@overload
def dict_to_geometry(geometry_dict: MultiPointDict) -> shapely.MultiPoint: ...


@overload
def dict_to_geometry(geometry_dict: LineStringDict) -> shapely.LineString: ...


@overload
def dict_to_geometry(geometry_dict: MultiLineStringDict) -> shapely.MultiLineString: ...


@overload
def dict_to_geometry(geometry_dict: PolygonDict) -> shapely.Polygon: ...


@overload
def dict_to_geometry(geometry_dict: MultiPolygonDict) -> shapely.MultiPolygon: ...


@overload
def dict_to_geometry(
    geometry_dict: GeometryCollectionDict,
) -> shapely.GeometryCollection: ...


def dict_to_geometry(geometry_dict: GeometryDict) -> BaseGeometry:
    """
    GeoJSON 的 geometry 字典转为几何对象

    See Also
    --------
    shapely.geometry.shape
    """
    return sgeom.shape(geometry_dict)  # pyright: ignore[reportArgumentType]


def make_feature(
    geometry_dict: GeometryDict, properties: Mapping[str, Any] | None = None
) -> FeatureDict:
    """用 geometry 和 properties 字典构造 GeoJSON 的 feature 字典"""
    if properties is None:
        properties = {}
    elif not isinstance(properties, dict):
        properties = dict(properties)

    return {"type": "Feature", "geometry": geometry_dict, "properties": properties}


def make_geojson(features: Iterable[FeatureDict]) -> GeoJSONDict:
    """用一组 feature 字典构造 GeoJSON 字典"""
    if not isinstance(features, list):
        features = list(features)
    return {"type": "FeatureCollection", "features": features}
