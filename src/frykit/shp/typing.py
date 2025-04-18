from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict, TypeVar, Union

import shapely
from shapely.geometry.base import BaseGeometry

PointCoordinates = Sequence[float]
MultiPointCoordinates = Sequence[PointCoordinates]
LineStringCoordinates = Sequence[PointCoordinates]
MultiLineStringCoordinates = Sequence[LineStringCoordinates]
PolygonCoordinates = Sequence[LineStringCoordinates]
MultiPolygonCoordinates = Sequence[PolygonCoordinates]


# TODO: 用 NotRequired 标注 bbox 字段
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


# 此处只能用 Union？
GeometryDict = Union[
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
    properties: dict


class GeoJSONDict(TypedDict):
    type: Literal["FeatureCollection"]
    features: list[FeatureDict]


PointType = shapely.Point | shapely.MultiPoint
LineStringType = shapely.LineString | shapely.MultiLineString
PolygonType = shapely.Polygon | shapely.MultiPolygon
GeometryT = TypeVar("GeometryT", bound=BaseGeometry)
