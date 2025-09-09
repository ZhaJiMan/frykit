from __future__ import annotations

from itertools import chain
from typing import Any, cast, overload

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
from frykit.utils import (
    deprecator,
    format_type_error,
    get_package_version,
    simple_deprecator,
)

__all__ = [
    "GeometryTransformer",
    "box_path",
    "dict_to_geometry",
    "geom_to_path",
    "geometry_to_dict",
    "geometry_to_shape",
    "get_geojson_geometries",
    "get_geojson_properties",
    "get_representative_xy",
    "get_shapefile_geometries",
    "get_shapefile_properties",
    "make_feature",
    "make_geojson",
    "orient_polygon",
    "path_to_polygon",
    "polygon_to_polys",
]


def _orient(polygon: shapely.Polygon, ccw: bool = True) -> shapely.Polygon:
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


_SHAPELY_2_1 = get_package_version("shapely") >= (2, 1, 0)


@overload
def orient_polygon(polygon: shapely.Polygon, ccw: bool = True) -> shapely.Polygon: ...


@overload
def orient_polygon(
    polygon: shapely.MultiPolygon, ccw: bool = True
) -> shapely.MultiPolygon: ...


def orient_polygon(polygon: PolygonType, ccw: bool = True) -> PolygonType:
    """调整多边形的绕行方向。例如 ccw=True 时外环逆时针，内环顺时针。"""
    if not isinstance(polygon, (shapely.Polygon, shapely.MultiPolygon)):
        raise TypeError(
            format_type_error(
                "polygon", polygon, [shapely.Polygon, shapely.MultiPolygon]
            )
        )

    if _SHAPELY_2_1:
        return shapely.orient_polygons(polygon, exterior_cw=not ccw)  # type: ignore

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
    """几何对象转为 GeoJSON 的 geometry 字典"""
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


def get_shapefile_properties(file_path: PathType) -> pd.DataFrame:
    """提取 shapefile 文件里的所有属性为 DataFrame"""
    with shapefile.Reader(file_path) as reader:
        records = [record.as_dict() for record in reader.iterRecords()]
    return pd.DataFrame.from_records(records)


def get_shapefile_geometries(file_path: PathType) -> list[BaseGeometry]:
    """提取 shapefile 文件里的所有几何对象"""
    with shapefile.Reader(file_path) as reader:
        return list(map(sgeom.shape, reader.iterShapes()))


def get_representative_xy(geometry: BaseGeometry) -> tuple[float, float]:
    """计算保证在几何对象内部的代表点"""
    point = geometry.representative_point()
    return point.x, point.y


def make_feature(
    geometry_dict: GeometryDict, properties: dict[str, Any] | None = None
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


@deprecator(alternative="frykit.plot.utils.geometry_to_path")
def geom_to_path(geom: BaseGeometry):
    from frykit.plot.utils import geometry_to_path

    return geometry_to_path(geom)


@deprecator(alternative="frykit.plot.utils.path_to_polygon")
def path_to_polygon(path) -> PolygonType:
    from frykit.plot.utils import path_to_polygon

    return path_to_polygon(path)


@deprecator(alternative="frykit.plot.utils.box_path")
def box_path(x0: float, x1: float, y0: float, y1: float):
    from frykit.plot.utils import box_path

    return box_path(x0, x1, y0, y1)


class GeometryTransformer:
    @simple_deprecator(
        "frykit.shp.utils.GeometryTransformer 已弃用，建议换用 frykit.plot.utils.project_geometry",
        raise_error=True,
    )
    def __init__(self, *args, **kwargs): ...
