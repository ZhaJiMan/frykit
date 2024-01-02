import struct
from io import BytesIO
from pathlib import PurePath
from typing import Any, Union

import numpy as np
import shapefile
import shapely.geometry as sgeom
from shapely.geometry.base import BaseGeometry

# 几何类型.
POLYGON_TYPE = 0
MULTI_POLYGON_TYPE = 1

# 数据类型.
DTYPE = '<I'
DTYPE_SIZE = 4

# 压缩参数.
LON0, LON1 = 70, 140
LAT0, LAT1 = 0, 60
N = DTYPE_SIZE * 8
ADD_OFFSETS = np.array([LON0, LAT0])
SCALE_FACTORS = np.array([LON1 - LON0, LAT1 - LAT0]) / (2**N - 1)

class BinaryConverter:
    '''将shapefile文件转为二进制文件的类.'''
    def __init__(self, filepath: Union[str, PurePath]) -> None:
        self.file = open(str(filepath), 'wb')

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> None:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def convert(self, filepath: Union[str, PurePath]) -> None:
        '''转换filepath指向的文件.'''
        with shapefile.Reader(str(filepath)) as reader:
            if reader.shapeType != 5:
                raise ValueError('shp文件必须是Polygon类型')
            geoj = reader.__geo_interface__
        content = self.pack_geoj(geoj)
        self.file.write(content)

    def pack_geoj(self, geoj: dict) -> bytes:
        '''将geojson对象打包成二进制.'''
        shapes = []
        shape_sizes = []
        for feature in geoj['features']:
            geometry = feature['geometry']
            if geometry['type'] == 'Polygon':
                shape = self.pack_polygon(geometry['coordinates'])
                shape_type = struct.pack(DTYPE, POLYGON_TYPE)
            elif geometry['type'] == 'MultiPolygon':
                shape = self.pack_multi_polygon(geometry['coordinates'])
                shape_type = struct.pack(DTYPE, MULTI_POLYGON_TYPE)
            else:
                raise NotImplementedError('不支持的几何类型')
            shape = shape_type + shape
            shape_sizes.append(len(shape))
            shapes.append(shape)

        shapes = b''.join(shapes)
        num_shapes = struct.pack(DTYPE, len(shape_sizes))
        shape_sizes = np.array(shape_sizes, DTYPE).tobytes()
        content = b''.join([num_shapes, shape_sizes, shapes])

        return content

    def pack_polygon(self, coordinates: list) -> bytes:
        '''将Polygon的坐标打包成二进制.'''
        rings = []
        ring_sizes = []
        for coords in coordinates:
            coords = np.array(coords)
            coords = np.round((coords - ADD_OFFSETS) / SCALE_FACTORS)
            ring = coords.astype(DTYPE).tobytes()
            ring_sizes.append(len(ring))
            rings.append(ring)

        rings = b''.join(rings)
        num_rings = struct.pack(DTYPE, len(ring_sizes))
        ring_sizes = np.array(ring_sizes, DTYPE).tobytes()
        polygon = b''.join([num_rings, ring_sizes, rings])

        return polygon

    def pack_multi_polygon(self, coordinates: list) -> bytes:
        '''将MultiPolygon的坐标打包成二进制.'''
        polygons = list(map(self.pack_polygon, coordinates))
        polygon_sizes = list(map(len, polygons))

        polygons = b''.join(polygons)
        num_polygons = struct.pack(DTYPE, len(polygon_sizes))
        polygon_sizes = np.array(polygon_sizes, DTYPE).tobytes()
        multi_polygon = b''.join([num_polygons, polygon_sizes, polygons])

        return multi_polygon

class BinaryReader:
    '''读取二进制文件的类.'''
    def __init__(self, filepath: Union[str, PurePath]) -> None:
        self.file = open(str(filepath), 'rb')
        self.num_shapes = struct.unpack(DTYPE, self.file.read(DTYPE_SIZE))[0]
        self.shape_sizes = np.frombuffer(
            self.file.read(self.num_shapes * DTYPE_SIZE), DTYPE
        )
        self.header_size = self.file.tell()
        self.shape_offsets = (
            self.shape_sizes.cumsum()
            - self.shape_sizes
            + self.header_size
        )

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> None:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def shape(self, i: int = 0) -> BaseGeometry:
        '''读取第i个几何对象.'''
        if i >= self.num_shapes:
            raise ValueError(f'i应该小于{self.num_shapes}')

        self.file.seek(self.shape_offsets[i])
        buffer = BytesIO(self.file.read(self.shape_sizes[i]))
        shape_type = struct.unpack(DTYPE, buffer.read(DTYPE_SIZE))[0]
        if shape_type == POLYGON_TYPE:
            return self.unpack_polygon(buffer)
        elif shape_type == MULTI_POLYGON_TYPE:
            return self.unpack_multi_polygon(buffer)
        else:
            raise NotImplementedError('不支持的几何类型')

    def shapes(self) -> list[BaseGeometry]:
        '''读取所有几何对象.'''
        shapes = []
        self.file.seek(self.header_size)
        for shape_size in self.shape_sizes:
            buffer = BytesIO(self.file.read(shape_size))
            shape_type = struct.unpack(DTYPE, buffer.read(DTYPE_SIZE))[0]
            if shape_type == POLYGON_TYPE:
                shape = self.unpack_polygon(buffer)
            elif shape_type == MULTI_POLYGON_TYPE:
                shape = self.unpack_multi_polygon(buffer)
            else:
                raise NotImplementedError('不支持的几何类型')
            shapes.append(shape)

        return shapes

    def unpack_polygon(self, buffer: BytesIO) -> sgeom.Polygon:
        '''将Polygon的二进制解包为几何对象.'''
        num_rings = struct.unpack(DTYPE, buffer.read(DTYPE_SIZE))[0]
        ring_sizes = np.frombuffer(buffer.read(num_rings * DTYPE_SIZE), DTYPE)

        rings = []
        for ring_size in ring_sizes:
            ring = np.frombuffer(buffer.read(ring_size), DTYPE).reshape(-1, 2)
            ring = ring * SCALE_FACTORS + ADD_OFFSETS
            rings.append(ring)
        polygon = sgeom.Polygon(rings[0], rings[1:])

        return polygon

    def unpack_multi_polygon(self, buffer: BytesIO) -> sgeom.MultiPolygon:
        '''将MultiPolygon的二进制解包为几何对象.'''
        num_polygons = struct.unpack(DTYPE, buffer.read(DTYPE_SIZE))[0]
        polygon_sizes = np.frombuffer(
            buffer.read(num_polygons * DTYPE_SIZE), DTYPE
        )

        polygons = []
        for polygon_size in polygon_sizes:
            buffer_ = BytesIO(buffer.read(polygon_size))
            polygon = self.unpack_polygon(buffer_)
            polygons.append(polygon)
        multi_polygon = sgeom.MultiPolygon(polygons)

        return multi_polygon

def convert_gcj_to_wgs(
    gcj_filepath: Union[str, PurePath],
    wgs_filepath: Union[str, PurePath],
    encoding: str = 'utf-8',
    validation: bool = True
) -> None:
    '''将GCJ-02坐标系的shapefile文件转为WGS84坐标系.'''
    from prcoords import gcj_wgs_bored
    with shapefile.Reader(str(gcj_filepath), encoding=encoding) as reader:
        with shapefile.Writer(str(wgs_filepath), encoding=encoding) as writer:
            writer.fields = reader.fields[1:]
            for shapeRec in reader.iterShapeRecords():
                writer.record(*shapeRec.record)
                shape = shapeRec.shape
                for i in range(len(shape.points)):
                    lon, lat = shape.points[i]
                    lat, lon = gcj_wgs_bored((lat, lon))
                    shape.points[i] = [lon, lat]
                if validation and not sgeom.shape(shape).is_valid:
                    raise ValueError('转换导致几何错误')
                writer.shape(shape)

def make_nine_line_file() -> None:
    '''根据阿里云的geojson制作九段线的shapefile.'''
    import json
    from frykit import DATA_DIRPATH
    geoj_filepath = DATA_DIRPATH / 'shp' / '100000_full.json'
    shp_filepath = DATA_DIRPATH / 'shp' / 'nine_line.shp'

    with open(str(geoj_filepath), encoding='utf-8') as f:
        geoj = json.load(f)
    geometry = geoj['features'][-1]['geometry']

    with shapefile.Writer(str(shp_filepath)) as writer:
        writer.fields = [['cn_adcode', 'C', 80, 0], ['cn_name', 'C', 80, 0]]
        writer.record(cn_adcode='100000', cn_name='九段线')
        writer.shape(geometry)