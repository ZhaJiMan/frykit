import struct
from io import BytesIO
from typing import Any

import numpy as np
import shapefile
import shapely.geometry as sgeom
from shapely.geometry.base import BaseGeometry

from frykit._typing import PathType

'''
利用类似 NetCDF 的有损压缩方式，将 64-bit 的 shapefile 转换成 32-bit 的整数。
高德地图数据的精度为 1e-6，压缩参数能保证 1e-7 的精度。大概够用了？
'''

# 几何类型
POINT = 0
MULTI_POINT = 1
LINE_STRING = 2
MULTI_LINE_STRING = 3
POLYGON = 4
MULTI_POLYGON = 5

# 数据类型
DTYPE = '<I'
DTYPE_SIZE = 4

# 压缩参数
LON0, LON1 = -180, 180
LAT0, LAT1 = -90, 90
N = DTYPE_SIZE * 8
ADD_OFFSETS = np.array([LON0, LAT0])
SCALE_FACTORS = np.array([LON1 - LON0, LAT1 - LAT0]) / (2**N - 1)


class BinaryPacker:
    '''将 shapefile 或 GeoJSON 的坐标数据打包成二进制的类'''

    def pack_shapefile(self, filepath: PathType) -> bytes:
        '''打包 shapefile'''
        with shapefile.Reader(str(filepath)) as reader:
            return self.pack_geojson(reader.__geo_interface__)

    def pack_geojson(self, geoj: dict) -> bytes:
        '''打包 GeoJSON'''
        shapes = []
        shape_sizes = []
        for feature in geoj['features']:
            shape = self.pack_geometry(feature['geometry'])
            shape_sizes.append(len(shape))
            shapes.append(shape)

        shapes = b''.join(shapes)
        num_shapes = struct.pack(DTYPE, len(shape_sizes))
        shape_sizes = np.array(shape_sizes, DTYPE).tobytes()
        content = num_shapes + shape_sizes + shapes

        return content

    def pack_geometry(self, geometry: dict) -> bytes:
        '''打包 GeoJSON 的 geometry 对象'''
        geometry_type = geometry['type']
        coordinates = geometry['coordinates']
        if geometry_type == 'Point':
            shape_data = self.pack_point(coordinates)
            shape_type = POINT
        elif geometry_type == 'MultiPoint':
            shape_data = self.pack_multi_point(coordinates)
            shape_type = MULTI_POINT
        elif geometry_type == 'LineString':
            shape_data = self.pack_line_string(coordinates)
            shape_type = LINE_STRING
        elif geometry_type == 'MultiLineString':
            shape_data = self.pack_multi_line_string(coordinates)
            shape_type = MULTI_LINE_STRING
        elif geometry_type == 'Polygon':
            shape_data = self.pack_polygon(coordinates)
            shape_type = POLYGON
        elif geometry_type == 'MultiPolygon':
            shape_data = self.pack_multi_polygon(coordinates)
            shape_type = MULTI_POLYGON
        else:
            raise TypeError(f'不支持的类型: {geometry_type}')
        shape_type = struct.pack(DTYPE, shape_type)
        shape = shape_type + shape_data

        return shape

    @staticmethod
    def pack_coords(coords: list) -> bytes:
        '''打包一维或二维的坐标数组'''
        coords = np.array(coords)
        coords = np.round((coords - ADD_OFFSETS) / SCALE_FACTORS)
        coords = coords.astype(DTYPE).tobytes()

        return coords

    def pack_point(self, coordinates: list) -> bytes:
        '''打包 Point 对象的坐标数据'''
        return self.pack_coords(coordinates)

    def pack_multi_point(self, coordinates: list) -> bytes:
        '''打包 MultiPoint 对象的坐标数据'''
        return self.pack_coords(coordinates)

    def pack_line_string(self, coordinates: list) -> bytes:
        '''打包 LineString 对象的坐标数据'''
        return self.pack_coords(coordinates)

    def pack_multi_line_string(self, coordinates: list) -> bytes:
        '''打包 MultiLineString 对象的坐标数据'''
        parts = list(map(self.pack_coords, coordinates))
        part_sizes = list(map(len, parts))
        parts = b''.join(parts)
        num_parts = struct.pack(DTYPE, len(coordinates))
        part_sizes = np.array(part_sizes, DTYPE).tobytes()
        data = num_parts + part_sizes + parts

        return data

    def pack_polygon(self, coordinates: list) -> bytes:
        '''打包 Polygon 对象的坐标数据'''
        return self.pack_multi_line_string(coordinates)

    def pack_multi_polygon(self, coordinates: list) -> bytes:
        '''打包 MultiPolygon 对象的坐标数据'''
        parts = list(map(self.pack_polygon, coordinates))
        part_sizes = list(map(len, parts))
        parts = b''.join(parts)
        num_parts = struct.pack(DTYPE, len(coordinates))
        part_sizes = np.array(part_sizes, DTYPE).tobytes()
        data = num_parts + part_sizes + parts

        return data


class BinaryReader:
    '''读取 BinaryPacker 类打包的二进制文件的类'''

    def __init__(self, filepath: PathType) -> None:
        self.file = open(str(filepath), 'rb')
        self.num_shapes = struct.unpack(DTYPE, self.file.read(DTYPE_SIZE))[0]
        self.shape_sizes = np.frombuffer(
            self.file.read(self.num_shapes * DTYPE_SIZE), DTYPE
        )
        self.header_size = self.file.tell()
        self.shape_offsets = (
            self.shape_sizes.cumsum() - self.shape_sizes + self.header_size
        )

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def shape(self, i: int = 0) -> BaseGeometry:
        '''读取第 i 个几何对象'''
        self.file.seek(self.shape_offsets[i])
        shape_type = struct.unpack(DTYPE, self.file.read(DTYPE_SIZE))[0]
        shape_data = self.file.read(self.shape_sizes[i] - DTYPE_SIZE)
        if shape_type == POINT:
            return self.unpack_point(shape_data)
        elif shape_type == MULTI_POINT:
            return self.unpack_multi_point(shape_data)
        elif shape_type == LINE_STRING:
            return self.unpack_line_string(shape_data)
        elif shape_type == MULTI_LINE_STRING:
            return self.unpack_multi_line_string(shape_data)
        elif shape_type == POLYGON:
            return self.unpack_polygon(shape_data)
        elif shape_type == MULTI_POLYGON:
            return self.unpack_multi_polygon(shape_data)
        else:
            raise TypeError(f'不支持的类型: {shape_type}')

    def shapes(self) -> list[BaseGeometry]:
        '''读取所有几何对象'''
        return list(map(self.shape, range(self.num_shapes)))

    @staticmethod
    def unpack_coords(coords: bytes) -> np.ndarray:
        '''解包一维或二维的坐标数组'''
        coords = np.frombuffer(coords, DTYPE).reshape(-1, 2)
        coords = coords * SCALE_FACTORS + ADD_OFFSETS

        return coords

    def unpack_point(self, data: bytes) -> sgeom.Point:
        '''解包 Point 对象的坐标数据'''
        return sgeom.Point(self.unpack_coords(data))

    def unpack_multi_point(self, data: bytes) -> sgeom.MultiPoint:
        '''解包 MultiPoint 对象的坐标数据'''
        return sgeom.MultiPoint(self.unpack_coords(data))

    def unpack_line_string(self, data: bytes) -> sgeom.LineString:
        '''解包 LineString 对象的坐标数据'''
        return sgeom.LineString(self.unpack_coords(data))

    def unpack_multi_line_string(self, data: bytes) -> sgeom.MultiLineString:
        '''解包 MultiLineString 对象的坐标数据'''
        with BytesIO(data) as f:
            num_parts = struct.unpack(DTYPE, f.read(DTYPE_SIZE))[0]
            part_sizes = np.frombuffer(f.read(num_parts * DTYPE_SIZE), DTYPE)
            lines = [self.unpack_coords(f.read(size)) for size in part_sizes]
            multi_line = sgeom.MultiLineString(lines)

        return multi_line

    def unpack_polygon(self, data: bytes) -> sgeom.Polygon:
        '''解包 Polygon 对象的坐标数据'''
        with BytesIO(data) as f:
            num_parts = struct.unpack(DTYPE, f.read(DTYPE_SIZE))[0]
            part_sizes = np.frombuffer(f.read(num_parts * DTYPE_SIZE), DTYPE)
            rings = [self.unpack_coords(f.read(size)) for size in part_sizes]
            polygon = sgeom.Polygon(rings[0], rings[1:])

        return polygon

    def unpack_multi_polygon(self, data: bytes) -> sgeom.MultiPolygon:
        '''解包 MultiPolygon 对象的坐标数据'''
        with BytesIO(data) as f:
            num_parts = struct.unpack(DTYPE, f.read(DTYPE_SIZE))[0]
            part_sizes = np.frombuffer(f.read(num_parts * DTYPE_SIZE), DTYPE)
            polygons = [
                self.unpack_polygon(f.read(size)) for size in part_sizes
            ]
            multi_polygon = sgeom.MultiPolygon(polygons)

        return multi_polygon
