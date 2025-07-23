from __future__ import annotations

import struct
from collections.abc import Sequence
from enum import IntEnum
from io import BytesIO
from typing import Any

import numpy as np
import shapely
from numpy.typing import ArrayLike, NDArray
from shapely.geometry.base import BaseGeometry

from frykit.calc import is_finite
from frykit.shp.typing import GeoJSONDict
from frykit.shp.utils import (
    get_geojson_geometries,
    get_shapefile_geometries,
    orient_polygon,
)
from frykit.typing import PathType

"""
- 用类似 NetCDF 的有损压缩方式，将 GeoJSON 的坐标数据转换成 uint32 或 uint16 的二进制。
高德地图数据精度为 1e-6，这里也指定压缩精度为 1e-6。
- 多边形按外环顺时针，内环逆时针的顺序保存。
"""

UINT32 = "<I"
INT16 = "<h"
UINT32_SIZE = struct.calcsize(UINT32)
INT16_SIZE = struct.calcsize(INT16)
UINT32_MIN = np.iinfo(UINT32).min
UINT32_MAX = np.iinfo(UINT32).max
INT16_MIN = np.iinfo(INT16).min
INT16_MAX = np.iinfo(INT16).max


class CoordsCodec:
    """用有损压缩对坐标做编码解码的类"""

    def __init__(self) -> None:
        self.lon0 = -180
        self.lon1 = 180
        self.lat0 = -90
        self.lat1 = 90

        self.precision = 1e-6
        self.add_offsets = np.array([self.lon0, self.lat0])
        self.scale_factors = 2 * np.array([self.precision, self.precision])

    def encode(self, coords: ArrayLike) -> bytes:
        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("要求 coords 是形如 (n, 2) 的二维数组")
        if not is_finite(coords):
            raise ValueError("coords 含有 NaN 或 Inf")

        coords = np.round((coords - self.add_offsets) / self.scale_factors)
        if coords.min() < 0 or coords.max() > UINT32_MAX:
            raise ValueError("coords 超出 uint32 压缩范围")
        diff_coords = np.diff(coords, axis=0)

        if (
            diff_coords.shape[0] > 0
            and diff_coords.min() >= INT16_MIN
            and diff_coords.max() <= INT16_MAX
        ):
            use_diff = True
            first_point = coords[0]
            binary = (
                first_point.astype(UINT32).tobytes()
                + diff_coords.astype(INT16).tobytes()
            )
        else:
            use_diff = False
            binary = coords.astype(UINT32).tobytes()

        header = struct.pack(UINT32, use_diff)
        binary = header + binary

        return binary

    def decode(self, binary: bytes) -> NDArray:
        with BytesIO(binary) as f:
            use_diff = bool(struct.unpack(UINT32, f.read(UINT32_SIZE))[0])
            if use_diff:
                first_point = np.frombuffer(f.read(2 * UINT32_SIZE), dtype=UINT32)
                diff_coords = np.frombuffer(f.read(), dtype=INT16).reshape(-1, 2)
                coords = np.vstack([first_point, diff_coords], dtype=int).cumsum(axis=0)
            else:
                coords = np.frombuffer(f.read(), dtype=UINT32).reshape(-1, 2)
        coords = coords.astype(float) * self.scale_factors + self.add_offsets

        return coords


_codec = CoordsCodec()


class GeometryEnum(IntEnum):
    Point = 0
    MultiPoint = 1
    LineString = 2
    MultiLineString = 3
    LinearRing = 4
    Polygon = 5
    MultiPolygon = 6
    GeometryCollection = 7


def _concat_binaries(binaries: list[bytes]) -> bytes:
    return (
        struct.pack(UINT32, len(binaries))
        + np.array(list(map(len, binaries)), dtype=UINT32).tobytes()
        + b"".join(binaries)
    )


def _split_binary(binary: bytes) -> list[bytes]:
    with BytesIO(binary) as f:
        num_binaries = struct.unpack(UINT32, f.read(UINT32_SIZE))[0]
        binary_sizes = np.frombuffer(f.read(num_binaries * UINT32_SIZE), dtype=UINT32)
        binaries = [f.read(size) for size in binary_sizes]
        return binaries


def _encode_point(point: shapely.Point) -> bytes:
    return _codec.encode(shapely.get_coordinates(point))


def _encode_multi_point(multi_point: shapely.MultiPoint) -> bytes:
    return _codec.encode(shapely.get_coordinates(multi_point))


def _encode_line_string(line_string: shapely.LineString) -> bytes:
    return _codec.encode(shapely.get_coordinates(line_string))


def _encode_multi_line_string(multi_line_string: shapely.MultiLineString) -> bytes:
    return _concat_binaries(list(map(_encode_line_string, multi_line_string.geoms)))


def _encode_polygon(polygon: shapely.Polygon) -> bytes:
    polygon = orient_polygon(polygon, ccw=False)
    linear_rings = [polygon.exterior, *polygon.interiors]
    return _concat_binaries(list(map(_encode_line_string, linear_rings)))


def _encode_multi_polygon(multi_polygon: shapely.MultiPolygon) -> bytes:
    return _concat_binaries(list(map(_encode_polygon, multi_polygon.geoms)))


def _encode_geometry(geometry: BaseGeometry) -> bytes:
    match geometry:
        case shapely.Point():
            binary = _encode_point(geometry)
        case shapely.MultiPoint():
            binary = _encode_multi_point(geometry)
        case shapely.LineString():
            binary = _encode_line_string(geometry)
        case shapely.MultiLineString():
            binary = _encode_multi_line_string(geometry)
        case shapely.Polygon():
            binary = _encode_polygon(geometry)
        case shapely.MultiPolygon():
            binary = _encode_multi_polygon(geometry)
        case shapely.GeometryCollection():
            binaries = list(map(_encode_geometry, geometry.geoms))
            binary = _concat_binaries(binaries)
        case _:
            raise ValueError(f"geometry_type: {geometry.geom_type}")

    header = struct.pack(UINT32, GeometryEnum[geometry.geom_type])
    binary = header + binary

    return binary


def _decode_point(binary: bytes) -> shapely.Point:
    x, y = _codec.decode(binary)[0]
    return shapely.Point(x, y)


def _decode_multi_point(binary: bytes) -> shapely.MultiPoint:
    return shapely.MultiPoint(_codec.decode(binary))


def _decode_line_string(binary: bytes) -> shapely.LineString:
    return shapely.LineString(_codec.decode(binary))


def _decode_multi_line_string(binary: bytes) -> shapely.MultiLineString:
    lines = list(map(_codec.decode, _split_binary(binary)))
    return shapely.MultiLineString(lines)


def _decode_linear_ring(binary: bytes) -> shapely.LinearRing:
    return shapely.LinearRing(_codec.decode(binary))


def _decode_polygon(binary: bytes) -> shapely.Polygon:
    coordinates = list(map(_codec.decode, _split_binary(binary)))
    return shapely.Polygon(coordinates[0], coordinates[1:])


def _decode_multi_polygon(binary: bytes) -> shapely.MultiPolygon:
    polygons = list(map(_decode_polygon, _split_binary(binary)))
    return shapely.MultiPolygon(polygons)


def _decode_geometry(binary: bytes) -> BaseGeometry:
    with BytesIO(binary) as f:
        enum_value = struct.unpack(UINT32, f.read(UINT32_SIZE))[0]
        geometry_type = GeometryEnum(enum_value).name
        binary = f.read()

    match geometry_type:
        case "Point":
            return _decode_point(binary)
        case "MultiPoint":
            return _decode_multi_point(binary)
        case "LineString":
            return _decode_line_string(binary)
        case "MultiLineString":
            return _decode_multi_line_string(binary)
        case "LinearRing":
            return _decode_linear_ring(binary)
        case "Polygon":
            return _decode_polygon(binary)
        case "MultiPolygon":
            return _decode_multi_polygon(binary)
        case "GeometryCollection":
            return shapely.GeometryCollection(
                list(map(_decode_geometry, _split_binary(binary)))
            )
        case _:
            raise ValueError(f"geometry_type: {geometry_type}")


def dump_geometries(geometries: Sequence[BaseGeometry]) -> bytes:
    """将一组几何对象编码成二进制"""
    return _concat_binaries(list(map(_encode_geometry, geometries)))


def dump_geojson(geojson_dict: GeoJSONDict) -> bytes:
    """将 GeoJSON 字典里的几何对象编码成二进制"""
    return dump_geometries(get_geojson_geometries(geojson_dict))


def dump_shapefile(file_path: PathType) -> bytes:
    """将 shapefile 文件里的几何对象编码成二进制"""
    return dump_geometries(get_shapefile_geometries(file_path))


def load_binary(binary: bytes) -> list[BaseGeometry]:
    """将二进制解码成一组几何对象"""
    return list(map(_decode_geometry, _split_binary(binary)))


class BinaryReader:
    """读取二进制文件的类"""

    def __init__(self, file_path: PathType) -> None:
        self.file = open(file_path, "rb")
        self.num_geometries = struct.unpack(UINT32, self.file.read(UINT32_SIZE))[0]
        self.binary_sizes = np.frombuffer(
            self.file.read(self.num_geometries * UINT32_SIZE), dtype=UINT32
        ).astype(int)
        self.header_size = self.file.tell()
        self.binary_offsets = (
            self.binary_sizes.cumsum() - self.binary_sizes + self.header_size
        )

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def geometry(self, i: int) -> BaseGeometry:
        """读取第 i 个几何对象"""
        self.file.seek(self.binary_offsets[i])
        return _decode_geometry(self.file.read(self.binary_sizes[i]))

    def geometries(self) -> list[BaseGeometry]:
        """读取所有几何对象"""
        self.file.seek(self.binary_offsets[0])
        return [_decode_geometry(self.file.read(size)) for size in self.binary_sizes]
