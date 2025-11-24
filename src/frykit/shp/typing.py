from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, Union

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    import shapely
    from shapely.geometry.base import BaseGeometry

__all__ = [
    "FeatureDict",
    "GeoJSONDict",
    "GeometryCollectionDict",
    "GeometryDict",
    "GeometryT",
    "LineStringCoordinates",
    "LineStringDict",
    "LineStringType",
    "MultiLineStringCoordinates",
    "MultiLineStringDict",
    "MultiPointCoordinates",
    "MultiPointDict",
    "MultiPolygonCoordinates",
    "MultiPolygonDict",
    "PointCoordinates",
    "PointDict",
    "PointType",
    "PolygonCoordinates",
    "PolygonDict",
    "PolygonType",
]

PointCoordinates: TypeAlias = list[float]
MultiPointCoordinates: TypeAlias = list[PointCoordinates]
LineStringCoordinates: TypeAlias = list[PointCoordinates]
MultiLineStringCoordinates: TypeAlias = list[LineStringCoordinates]
PolygonCoordinates: TypeAlias = list[LineStringCoordinates]
MultiPolygonCoordinates: TypeAlias = list[PolygonCoordinates]


class PointDict(TypedDict, extra_items=Any):
    type: Literal["Point"]
    coordinates: PointCoordinates
    bbox: NotRequired[list[float]]


class MultiPointDict(TypedDict, extra_items=Any):
    type: Literal["MultiPoint"]
    coordinates: MultiPointCoordinates
    bbox: NotRequired[list[float]]


class LineStringDict(TypedDict, extra_items=Any):
    type: Literal["LineString"]
    coordinates: LineStringCoordinates
    bbox: NotRequired[list[float]]


class MultiLineStringDict(TypedDict, extra_items=Any):
    type: Literal["MultiLineString"]
    coordinates: MultiLineStringCoordinates
    bbox: NotRequired[list[float]]


class PolygonDict(TypedDict, extra_items=Any):
    type: Literal["Polygon"]
    coordinates: PolygonCoordinates
    bbox: NotRequired[list[float]]


class MultiPolygonDict(TypedDict, extra_items=Any):
    type: Literal["MultiPolygon"]
    coordinates: MultiPolygonCoordinates
    bbox: NotRequired[list[float]]


# X | Y 不支持部分前向引用
# https://docs.python.org/3/library/stdtypes.html#union-type
GeometryDict: TypeAlias = Union[
    PointDict,
    MultiPointDict,
    LineStringDict,
    MultiLineStringDict,
    PolygonDict,
    MultiPolygonDict,
    "GeometryCollectionDict",
]


class GeometryCollectionDict(TypedDict, extra_items=Any):
    type: Literal["GeometryCollection"]
    geometries: list[GeometryDict]
    bbox: NotRequired[list[float]]


class FeatureDict(TypedDict, extra_items=Any):
    type: Literal["Feature"]
    geometry: GeometryDict
    properties: dict[str, Any]
    bbox: NotRequired[list[float]]


class GeoJSONDict(TypedDict, extra_items=Any):
    type: Literal["FeatureCollection"]
    features: list[FeatureDict]
    bbox: NotRequired[list[float]]


# 赋值操作需要前向引用
PointType: TypeAlias = "shapely.Point | shapely.MultiPoint"
LineStringType: TypeAlias = "shapely.LineString | shapely.MultiLineString"
PolygonType: TypeAlias = "shapely.Polygon | shapely.MultiPolygon"
GeometryT = TypeVar("GeometryT", bound="BaseGeometry")
