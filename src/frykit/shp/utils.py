from __future__ import annotations

from itertools import chain
from typing import cast, overload

import pandas as pd
import shapefile
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
from frykit.typing import PathType
from frykit.utils import deprecator, format_type_error

__all__ = [
    "orient_polygon",
    "geometry_to_shape",
    "geometry_to_dict",
    "dict_to_geometry",
    "get_geojson_properties",
    "get_geojson_geometries",
    "get_shapefile_properties",
    "get_shapefile_geometries",
    "get_representative_xy",
    "make_point_dict",
    "make_multi_point_dict",
    "make_line_string_dict",
    "make_multi_line_string_dict",
    "make_polygon_dict",
    "make_multi_polygon_dict",
    "make_geometry_collection_dict",
    "make_feature",
    "make_geojson",
    "polygon_to_polys",
    "geom_to_path",
    "path_to_polygon",
    "box_path",
    "GeometryTransformer",
]


def _orient(polygon: shapely.Polygon, ccw: bool = True) -> shapely.Polygon:
    """调整 shapely.Polygon 的绕行方向"""
    if polygon.is_empty:
        return polygon

    # 循环比向量化略快？
    linear_rings = []
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


# TODO: shapely.orient_polygons in 2.1.0
def orient_polygon(polygon: PolygonType, ccw: bool = True) -> PolygonType:
    """调整多边形的绕行方向。例如 ccw=True 时外环逆时针，内环顺时针。"""
    match polygon:
        case shapely.Polygon():
            return _orient(polygon, ccw)
        case shapely.MultiPolygon():
            return shapely.MultiPolygon([_orient(part, ccw) for part in polygon.geoms])
        case _:
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

    def get_coordinates(geometry: BaseGeometry) -> list[list[float]]:
        return shapely.get_coordinates(geometry).tolist()

    def polygon_to_shape(polygon: shapely.Polygon) -> PolygonCoordinates:
        polygon = _orient(polygon, ccw=False)
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
    """GeoJSON 的 geometry 字典转为几何对象"""
    return sgeom.shape(geometry_dict)  # type: ignore


def get_geojson_properties(geojson_dict: GeoJSONDict) -> pd.DataFrame:
    """提取 GeoJSON 字典里的所有 properties 为 DataFrame"""
    records = [feature["properties"] for feature in geojson_dict["features"]]
    return pd.DataFrame.from_records(records)


def get_geojson_geometries(geojson_dict: GeoJSONDict) -> list[BaseGeometry]:
    """提取 GeoJSON 字典里的所有几何对象"""
    return [
        dict_to_geometry(feature["geometry"]) for feature in geojson_dict["features"]
    ]


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


def make_point_dict(coordinates: PointCoordinates) -> PointDict:
    """构造 Point 字典"""
    return {"type": "Point", "coordinates": coordinates}


def make_multi_point_dict(coordinates: MultiPointCoordinates) -> MultiPointDict:
    """构造 MultiPoint 字典"""
    return {"type": "MultiPoint", "coordinates": coordinates}


def make_line_string_dict(coordinates: LineStringCoordinates) -> LineStringDict:
    """构造 LineString 字典"""
    return {"type": "LineString", "coordinates": coordinates}


def make_multi_line_string_dict(
    coordinates: MultiLineStringCoordinates,
) -> MultiLineStringDict:
    """构造 MultiLineString 字典"""
    return {"type": "MultiLineString", "coordinates": coordinates}


def make_polygon_dict(coordinates: PolygonCoordinates) -> PolygonDict:
    """构造 Polygon 字典"""
    return {"type": "Polygon", "coordinates": coordinates}


def make_multi_polygon_dict(coordinates: MultiPolygonCoordinates) -> MultiPolygonDict:
    """构造 MultiPolygon 字典"""
    return {"type": "MultiPolygon", "coordinates": coordinates}


def make_geometry_collection_dict(
    geometries: list[GeometryDict],
) -> GeometryCollectionDict:
    """构造 GeometryCollection 字典"""
    return {"type": "GeometryCollection", "geometries": geometries}


def make_feature(
    geometry_dict: GeometryDict, properties: dict | None = None
) -> FeatureDict:
    """用 geometry 和 properties 字典构造 GeoJSON 的 feature 字典"""
    feature = {"type": "Feature", "geometry": geometry_dict}
    if properties is not None:
        feature["properties"] = properties

    return feature  # type: ignore


def make_geojson(features: list[FeatureDict]) -> GeoJSONDict:
    """用一组 feature 字典构造 GeoJSON 字典"""
    return {"type": "FeatureCollection", "features": features}


@deprecator(alternative=geometry_to_shape)
def polygon_to_polys(polygon: PolygonType) -> PolygonCoordinates:
    return geometry_to_shape(polygon)


@deprecator(alternative="frykit.plot.geometry_to_path")
def geom_to_path(geom: BaseGeometry):
    from frykit.plot.utils import geometry_to_path

    return geometry_to_path(geom)


@deprecator(alternative="frykit.plot.path_to_polygon")
def path_to_polygon(path) -> PolygonType:
    from frykit.plot.utils import path_to_polygon

    return path_to_polygon(path)


@deprecator(alternative="frykit.plot.box_path")
def box_path(x0: float, x1: float, y0: float, y1: float):
    from frykit.plot.utils import box_path

    return box_path(x0, x1, y0, y1)


class GeometryTransformer:
    @deprecator(alternative="frykit.plot.project_geometry", raise_error=True)
    def __init__(self, *args, **kwargs): ...
