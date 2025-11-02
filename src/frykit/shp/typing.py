from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict, TypeVar, Union

from typing_extensions import NotRequired

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


class PointDict(TypedDict):
    type: Literal["Point"]
    coordinates: PointCoordinates


class MultiPointDict(TypedDict):
    type: Literal["MultiPoint"]
    coordinates: MultiPointCoordinates


class LineStringDict(TypedDict):
    type: Literal["LineString"]
    coordinates: LineStringCoordinates


class MultiLineStringDict(TypedDict):
    type: Literal["MultiLineString"]
    coordinates: MultiLineStringCoordinates


class PolygonDict(TypedDict):
    type: Literal["Polygon"]
    coordinates: PolygonCoordinates


class MultiPolygonDict(TypedDict):
    type: Literal["MultiPolygon"]
    coordinates: MultiPolygonCoordinates


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


class GeometryCollectionDict(TypedDict):
    type: Literal["GeometryCollection"]
    geometries: list[GeometryDict]


class FeatureDict(TypedDict):
    type: Literal["Feature"]
    geometry: GeometryDict
    properties: dict[str, Any]
    bbox: NotRequired[list[float]]


class GeoJSONDict(TypedDict):
    type: Literal["FeatureCollection"]
    features: list[FeatureDict]
    bbox: NotRequired[list[float]]


# 赋值操作需要前向引用
PointType: TypeAlias = "shapely.Point | shapely.MultiPoint"
LineStringType: TypeAlias = "shapely.LineString | shapely.MultiLineString"
PolygonType: TypeAlias = "shapely.Polygon | shapely.MultiPolygon"
GeometryT = TypeVar("GeometryT", bound="BaseGeometry")
