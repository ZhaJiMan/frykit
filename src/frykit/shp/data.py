from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, cast, overload

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray

if TYPE_CHECKING:
    import geopandas as gpd

from frykit import get_data_dir
from frykit.conf import DataSource, config
from frykit.shp.binary import BinaryReader
from frykit.shp.typing import LineStringType, PolygonType
from frykit.utils import deprecator, format_type_error

__all__ = [
    "AdminLevel",
    "CityProperties",
    "DistrictProperties",
    "LineEnum",
    "LineName",
    "NameOrAdcode",
    "Properties",
    "ProvinceProperties",
    "clear_data_cache",
    "clear_data_cache",
    "get_cn_border",
    "get_cn_city",
    "get_cn_city_dataframe",
    "get_cn_city_geodataframe",
    "get_cn_city_names",
    "get_cn_city_properties",
    "get_cn_city_table",
    "get_cn_district",
    "get_cn_district_dataframe",
    "get_cn_district_geodataframe",
    "get_cn_district_names",
    "get_cn_district_properties",
    "get_cn_district_table",
    "get_cn_line",
    "get_cn_province",
    "get_cn_province_dataframe",
    "get_cn_province_geodataframe",
    "get_cn_province_names",
    "get_cn_province_properties",
    "get_cn_province_table",
    "get_countries",
    "get_land",
    "get_nine_line",
    "get_ocean",
]


def _resolve_data_source(data_source: DataSource | None) -> DataSource:
    if data_source is None:
        return config.data_source
    else:
        config.validate("data_source", data_source)
        return data_source


def _get_china_dir() -> Path:
    return get_data_dir() / "china"


def _get_world_dir() -> Path:
    return get_data_dir() / "world"


AdminLevel: TypeAlias = Literal["province", "city", "district"]


@cache
def _get_cn_table(level: AdminLevel, data_source: DataSource) -> pd.DataFrame:
    filepath = _get_china_dir() / data_source / f"cn_{level}.csv"
    return pd.read_csv(filepath)


def _get_cn_province_table(data_source: DataSource | None = None) -> pd.DataFrame:
    data_source = _resolve_data_source(data_source)
    return _get_cn_table("province", data_source)


def _get_cn_city_table(data_source: DataSource | None = None) -> pd.DataFrame:
    data_source = _resolve_data_source(data_source)
    return _get_cn_table("city", data_source)


def _get_cn_district_table(data_source: DataSource | None = None) -> pd.DataFrame:
    data_source = _resolve_data_source(data_source)
    return _get_cn_table("district", data_source)


def get_cn_province_table(data_source: DataSource | None = None) -> pd.DataFrame:
    """获取中国省界元数据的表格"""
    return _get_cn_province_table(data_source).copy()


def get_cn_city_table(data_source: DataSource | None = None) -> pd.DataFrame:
    """获取中国市界元数据的表格"""
    return _get_cn_city_table(data_source).copy()


def get_cn_district_table(data_source: DataSource | None = None) -> pd.DataFrame:
    """获取中国县界元数据的表格"""
    return _get_cn_district_table(data_source).copy()


NameOrAdcode: TypeAlias = int | str


def _get_cn_indices(
    name_to_indices: dict[str, NDArray[np.int64]],
    adcode_to_indices: dict[np.int64, NDArray[np.int64]],
    key: NameOrAdcode | Iterable[NameOrAdcode],
) -> NDArray[np.int64]:
    if isinstance(key, str) or not isinstance(key, Iterable):
        keys = [key]
    else:
        keys = key

    arrs: list[NDArray[np.int64]] = []
    for k in keys:
        match k:
            case str():
                arrs.append(name_to_indices[k])
            case int() | np.integer():
                arrs.append(adcode_to_indices[cast(np.int64, k)])
            case _:
                raise TypeError(format_type_error("key", k, [str, int]))

    return np.concatenate(arrs)


@dataclass
class Lookup:
    index: NDArray[np.int64]


@dataclass
class ProvinceLookup(Lookup):
    province_name: dict[str, NDArray[np.int64]]
    province_adcode: dict[np.int64, NDArray[np.int64]]


@dataclass
class CityLookup(Lookup):
    province_name: dict[str, NDArray[np.int64]]
    province_adcode: dict[np.int64, NDArray[np.int64]]
    city_name: dict[str, NDArray[np.int64]]
    city_adcode: dict[np.int64, NDArray[np.int64]]


@dataclass
class DistrictLookup(Lookup):
    province_name: dict[str, NDArray[np.int64]]
    province_adcode: dict[np.int64, NDArray[np.int64]]
    city_name: dict[str, NDArray[np.int64]]
    city_adcode: dict[np.int64, NDArray[np.int64]]
    district_name: dict[str, NDArray[np.int64]]
    district_adcode: dict[np.int64, NDArray[np.int64]]


@overload
def _get_cn_lookup(
    admin_level: Literal["province"], data_source: DataSource
) -> ProvinceLookup: ...


@overload
def _get_cn_lookup(
    admin_level: Literal["city"], data_source: DataSource
) -> CityLookup: ...


@overload
def _get_cn_lookup(
    admin_level: Literal["district"], data_source: DataSource
) -> DistrictLookup: ...


@cache
def _get_cn_lookup(
    admin_level: AdminLevel, data_source: DataSource
) -> ProvinceLookup | CityLookup | DistrictLookup:
    df = _get_cn_table(admin_level, data_source)
    index = df.index.to_numpy()
    match admin_level:
        case "province":
            cls = ProvinceLookup
            cols = df.columns[:2]
        case "city":
            cls = CityLookup
            cols = df.columns[:4]
        case "district":
            cls = DistrictLookup
            cols = df.columns[:6]

    # 第一次运行 groupby 略慢
    return cls(index, *[df.groupby(col).indices for col in cols])  # type: ignore


def _get_cn_province_lookup(data_source: DataSource | None = None) -> ProvinceLookup:
    data_source = _resolve_data_source(data_source)
    return _get_cn_lookup("province", data_source)


def _get_cn_city_lookup(data_source: DataSource | None = None) -> CityLookup:
    data_source = _resolve_data_source(data_source)
    return _get_cn_lookup("city", data_source)


def _get_cn_district_lookup(data_source: DataSource | None = None) -> DistrictLookup:
    data_source = _resolve_data_source(data_source)
    return _get_cn_lookup("district", data_source)


def _get_cn_province_indices(
    province: NameOrAdcode | Iterable[NameOrAdcode] | None,
    data_source: DataSource | None = None,
) -> NDArray[np.int64]:
    lookup = _get_cn_province_lookup(data_source)
    if province is not None:
        return _get_cn_indices(lookup.province_name, lookup.province_adcode, province)
    else:
        return lookup.index


def _get_cn_city_indices(
    city: NameOrAdcode | Iterable[NameOrAdcode] | None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None,
    data_source: DataSource | None = None,
) -> NDArray[np.int64]:
    lookup = _get_cn_city_lookup(data_source)
    if city is None and province is None:
        return lookup.index
    if city is not None and province is not None:
        raise ValueError("不能同时指定 city 和 province")

    if city is not None:
        return _get_cn_indices(lookup.city_name, lookup.city_adcode, city)
    else:
        assert province is not None
        return _get_cn_indices(lookup.province_name, lookup.province_adcode, province)


def _get_cn_district_indices(
    district: NameOrAdcode | Iterable[NameOrAdcode] | None,
    city: NameOrAdcode | Iterable[NameOrAdcode] | None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None,
    data_source: DataSource | None = None,
) -> NDArray[np.int64]:
    lookup = _get_cn_district_lookup(data_source)
    num_keys = sum(key is not None for key in [district, city, province])
    if num_keys == 0:
        return lookup.index
    if num_keys >= 2:
        raise ValueError("district、city 和 province 三个参数中只能指定一个")

    if district is not None:
        indices = _get_cn_indices(
            lookup.district_name, lookup.district_adcode, district
        )
    elif city is not None:
        indices = _get_cn_indices(lookup.city_name, lookup.city_adcode, city)
    else:
        assert province is not None
        indices = _get_cn_indices(
            lookup.province_name, lookup.province_adcode, province
        )

    if isinstance(district, str) and len(indices) > 1:
        df = _get_cn_district_table(data_source)
        df_str = df.iloc[indices, :6].to_string(index=False)
        raise ValueError(f"存在复数个同名的县，请用 adcode 指定\n{df_str}")

    return indices


# TypedDict 子类不需要多重继承
class Properties(TypedDict):
    short_name: str
    lon: float
    lat: float


class ProvinceProperties(Properties):
    province_name: str
    province_adcode: int


class CityProperties(Properties):
    province_name: str
    province_adcode: int
    city_name: str
    city_adcode: int


class DistrictProperties(Properties):
    province_name: str
    province_adcode: int
    city_name: str
    city_adcode: int
    district_name: str
    district_adcode: int


@overload
def get_cn_province_properties(
    province: NameOrAdcode, data_source: DataSource | None = None
) -> ProvinceProperties: ...


@overload
def get_cn_province_properties(
    province: Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> list[ProvinceProperties]: ...


def get_cn_province_properties(
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> ProvinceProperties | list[ProvinceProperties]:
    """
    获取中国省界的元数据

    Parameters
    ----------
    province : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        省名或 adcode。可以是多个省。默认为 None，表示所有省。

    data_source : {'amap', 'tianditu'} or None, default None
        数据源。默认为 None，表示使用默认的全局配置 'amap'。

    Returns
    -------
    ProvinceProperties or list of ProvinceProperties
        元数据字典
    """
    df = _get_cn_province_table(data_source)
    indices = _get_cn_province_indices(province, data_source)
    if len(indices) != len(df):
        df = df.iloc[indices]
    result = df.to_dict(orient="records")
    result = cast(list[ProvinceProperties], result)

    if isinstance(province, (str, int)):
        return result[0]
    else:
        return result


@overload
def get_cn_city_properties(
    city: NameOrAdcode,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> CityProperties: ...


@overload
def get_cn_city_properties(
    city: Iterable[NameOrAdcode] | None = None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> list[CityProperties]: ...


def get_cn_city_properties(
    city: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> CityProperties | list[CityProperties]:
    """
    获取中国市界的元数据

    Parameters
    ----------
    city : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        市名或 adcode。可以是多个市。默认为 None，表示所有市。

    province : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        省名或 adcode。表示指定某个省的所有市。可以是多个省。
        默认为 None，表示不指定省。

    data_source : {'amap', 'tianditu'} or None, default None
        数据源。默认为 None，表示使用默认的全局配置 'amap'。

    Returns
    -------
    CityProperties or list of CityProperties
        元数据字典
    """
    df = _get_cn_city_table(data_source)
    indices = _get_cn_city_indices(city, province, data_source)
    if len(indices) != len(df):
        df = df.iloc[indices]
    result = df.to_dict(orient="records")
    result = cast(list[CityProperties], result)

    if isinstance(city, (str, int)):
        return result[0]
    else:
        return result


@overload
def get_cn_district_properties(
    district: NameOrAdcode,
    city: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> DistrictProperties: ...


@overload
def get_cn_district_properties(
    district: Iterable[NameOrAdcode] | None = None,
    city: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> list[DistrictProperties]: ...


def get_cn_district_properties(
    district: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    city: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> DistrictProperties | list[DistrictProperties]:
    """
    获取中国县界的元数据

    Parameters
    ----------
    district : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        县名或 adcode。可以是多个县。默认为 None，表示所有县。

    city : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        市名或 adcode。表示指定某个市的所有县。可以是多个市。
        默认为 None，表示不指定市。

    province : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        省名或 adcode。表示指定某个省的所有县。可以是多个省。
        默认为 None，表示不指定省。

    data_source : {'amap', 'tianditu'} or None, default None
        数据源。默认为 None，表示使用默认的全局配置 'amap'。

    Returns
    -------
    DistrictProperties or list of DistrictProperties
        元数据字典
    """
    df = _get_cn_district_table(data_source)
    indices = _get_cn_district_indices(district, city, province, data_source)
    if len(indices) != len(df):
        df = df.iloc[indices]
    result = df.to_dict(orient="records")
    result = cast(list[DistrictProperties], result)

    if isinstance(district, (str, int)):
        return result[0]
    else:
        return result


def get_cn_province_names(
    short_name: bool = False, data_source: DataSource | None = None
) -> list[str]:
    """获取中国所有省的名字"""
    df = _get_cn_province_table(data_source)
    key = "short_name" if short_name else "province_name"
    return df[key].tolist()


def get_cn_city_names(
    short_name: bool = False, data_source: DataSource | None = None
) -> list[str]:
    """获取中国所有市的名字"""
    df = _get_cn_city_table(data_source)
    key = "short_name" if short_name else "city_name"
    return df[key].tolist()


def get_cn_district_names(
    short_name: bool = False, data_source: DataSource | None = None
) -> list[str]:
    """获取中国所有县的名字"""
    df = _get_cn_district_table(data_source)
    key = "short_name" if short_name else "district_name"
    return df[key].tolist()


@cache
def _get_cn_polygons(level: AdminLevel, data_source: DataSource) -> list[PolygonType]:
    filepath = _get_china_dir() / data_source / f"cn_{level}.bin"
    with BinaryReader(filepath) as reader:
        polygons = reader.geometries()
        return cast(list[PolygonType], polygons)


def _get_cn_province_polygons(
    data_source: DataSource | None = None,
) -> list[PolygonType]:
    data_source = _resolve_data_source(data_source)
    return _get_cn_polygons("province", data_source)


def _get_cn_city_polygons(data_source: DataSource | None = None) -> list[PolygonType]:
    data_source = _resolve_data_source(data_source)
    return _get_cn_polygons("city", data_source)


def _get_cn_district_polygons(
    data_source: DataSource | None = None,
) -> list[PolygonType]:
    data_source = _resolve_data_source(data_source)
    return _get_cn_polygons("district", data_source)


@overload
def get_cn_province(
    province: NameOrAdcode, data_source: DataSource | None = None
) -> PolygonType: ...


@overload
def get_cn_province(
    province: Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> list[PolygonType]: ...


def get_cn_province(
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> PolygonType | list[PolygonType]:
    """
    获取中国省界的多边形

    Parameters
    ----------
    province : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        省名或 adcode。可以是多个省。默认为 None，表示所有省。

    data_source : {'amap', 'tianditu'} or None, default None
        数据源。默认为 None，表示使用默认的全局配置 'amap'。

    Returns
    -------
    Polygon or MultiPolygon or list of Polygon and MultiPolygon
        多边形对象
    """
    indices = _get_cn_province_indices(province, data_source)
    polygons = _get_cn_province_polygons(data_source)
    if len(indices) == len(polygons):
        return polygons

    polygons = [polygons[i] for i in indices]
    if isinstance(province, (str, int)):
        return polygons[0]
    else:
        return polygons


@overload
def get_cn_city(
    city: NameOrAdcode,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> PolygonType: ...


@overload
def get_cn_city(
    city: Iterable[NameOrAdcode] | None = None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> list[PolygonType]: ...


def get_cn_city(
    city: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> PolygonType | list[PolygonType]:
    """
    获取中国市界的多边形

    Parameters
    ----------
    city : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        市名或 adcode。可以是多个市。默认为 None，表示所有市。

    province : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        省名或 adcode。表示指定某个省的所有市。可以是多个省。
        默认为 None，表示不指定省。

    data_source : {'amap', 'tianditu'} or None, default None
        数据源。默认为 None，表示使用默认的全局配置 'amap'。

    Returns
    -------
    Polygon or MultiPolygon or list of Polygon and MultiPolygon
        多边形对象
    """
    indices = _get_cn_city_indices(city, province, data_source)
    polygons = _get_cn_city_polygons(data_source)
    if len(indices) == len(polygons):
        return polygons

    polygons = [polygons[i] for i in indices]
    if isinstance(city, (str, int)):
        return polygons[0]
    else:
        return polygons


@overload
def get_cn_district(
    district: NameOrAdcode,
    city: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> PolygonType: ...


@overload
def get_cn_district(
    district: Iterable[NameOrAdcode] | None = None,
    city: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> list[PolygonType]: ...


def get_cn_district(
    district: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    city: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None = None,
    data_source: DataSource | None = None,
) -> PolygonType | list[PolygonType]:
    """
    获取中国县界的多边形

    Parameters
    ----------
    district : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        县名或 adcode。可以是多个县。默认为 None，表示所有县。

    city : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        市名或 adcode。表示指定某个市的所有县。可以是多个市。
        默认为 None，表示不指定市。

    province : NameOrAdcode or iterable object of NameOrAdcode or None, default None
        省名或 adcode。表示指定某个省的所有县。可以是多个省。
        默认为 None，表示不指定省。

    data_source : {'amap', 'tianditu'} or None, default None
        数据源。默认为 None，表示使用默认的全局配置 'amap'。

    Returns
    -------
    Polygon or MultiPolygon or list of Polygon and MultiPolygon
        多边形对象
    """
    indices = _get_cn_district_indices(district, city, province, data_source)
    polygons = _get_cn_district_polygons(data_source)
    if len(indices) == len(polygons):
        return polygons

    polygons = [polygons[i] for i in indices]
    if isinstance(district, (str, int)):
        return polygons[0]
    else:
        return polygons


@cache
def _get_cn_border(data_source: DataSource) -> shapely.MultiPolygon:
    filepath = _get_china_dir() / data_source / "cn_border.bin"
    with BinaryReader(filepath) as reader:
        return cast(shapely.MultiPolygon, reader.geometry(0))


def get_cn_border(data_source: DataSource | None = None) -> shapely.MultiPolygon:
    """获取中国国界的多边形"""
    data_source = _resolve_data_source(data_source)
    return _get_cn_border(data_source)


@cache
def _get_cn_line_strings() -> list[LineStringType]:
    filepath = _get_china_dir() / "cn_line.bin"
    with BinaryReader(filepath) as reader:
        line_strings = reader.geometries()
        return cast(list[LineStringType], line_strings)


LineName: TypeAlias = Literal["省界", "特别行政区界", "九段线", "未定国界"]


class LineEnum(IntEnum):
    省界 = 0
    特别行政区界 = 1
    九段线 = 2
    未定国界 = 3


@overload
def get_cn_line(name: LineName = "九段线") -> LineStringType: ...


@overload
def get_cn_line(name: Iterable[LineName]) -> list[LineStringType]: ...


def get_cn_line(
    name: LineName | Iterable[LineName] = "九段线",
) -> LineStringType | list[LineStringType]:
    """
    获取中国的修饰线段

    Parameters
    ----------
    name : {'省界', '特别行政区界', '九段线', '未定国界'} or iterable object of str, default '九段线'
        线段名称。可以是多种线段。默认为 '九段线'。

    Returns
    -------
    LineString or MultiLineString or list of LineString and MultiLineString
        线段对象
    """
    line_strings = _get_cn_line_strings()
    if isinstance(name, str):
        return line_strings[LineEnum[name]]
    else:
        return [line_strings[LineEnum[n]] for n in name]


@cache
def get_countries() -> list[PolygonType]:
    """获取所有国界的多边形"""
    filepath = _get_world_dir() / "country.bin"
    with BinaryReader(filepath) as reader:
        return cast(list[PolygonType], reader.geometries())


@cache
def get_land() -> shapely.MultiPolygon:
    """获取陆地多边形"""
    filepath = _get_world_dir() / "land.bin"
    with BinaryReader(filepath) as reader:
        return cast(shapely.MultiPolygon, reader.geometry(0))


@cache
def get_ocean() -> shapely.MultiPolygon:
    """获取海洋多边形"""
    filepath = _get_world_dir() / "ocean.bin"
    with BinaryReader(filepath) as reader:
        return cast(shapely.MultiPolygon, reader.geometry(0))


@cache
def _get_cn_dataframe(level: AdminLevel, data_source: DataSource) -> pd.DataFrame:
    df = _get_cn_table(level, data_source)
    polygons = _get_cn_polygons(level, data_source)
    return df.assign(geometry=polygons)


def get_cn_province_dataframe(data_source: DataSource | None = None) -> pd.DataFrame:
    """获取中国省界的多边形和元数据的 DataFrame"""
    data_source = _resolve_data_source(data_source)
    return _get_cn_dataframe("province", data_source).copy()


def get_cn_city_dataframe(data_source: DataSource | None = None) -> pd.DataFrame:
    """获取中国市界的多边形和元数据的 DataFrame"""
    data_source = _resolve_data_source(data_source)
    return _get_cn_dataframe("city", data_source).copy()


def get_cn_district_dataframe(data_source: DataSource | None = None) -> pd.DataFrame:
    """获取中国县界的多边形和元数据的 DataFrame"""
    data_source = _resolve_data_source(data_source)
    return _get_cn_dataframe("district", data_source).copy()


@cache
def _get_cn_geodataframe(
    level: AdminLevel, data_source: DataSource
) -> gpd.GeoDataFrame:
    import geopandas as gpd

    df = _get_cn_table(level, data_source)
    polygons = _get_cn_polygons(level, data_source)
    return gpd.GeoDataFrame(df, geometry=polygons, crs="EPSG:4326", copy=True)


def get_cn_province_geodataframe(
    data_source: DataSource | None = None,
) -> gpd.GeoDataFrame:
    """获取中国省界的多边形和元数据的 GeoDataFrame"""
    data_source = _resolve_data_source(data_source)
    return _get_cn_geodataframe("province", data_source)


def get_cn_city_geodataframe(data_source: DataSource | None = None) -> gpd.GeoDataFrame:
    """获取中国市界的多边形和元数据的 GeoDataFrame"""
    data_source = _resolve_data_source(data_source)
    return _get_cn_geodataframe("city", data_source)


def get_cn_district_geodataframe(
    data_source: DataSource | None = None,
) -> gpd.GeoDataFrame:
    """获取中国县界的多边形和元数据的 GeoDataFrame"""
    data_source = _resolve_data_source(data_source)
    return _get_cn_geodataframe("district", data_source)


def clear_data_cache() -> None:
    """清除数据缓存"""
    _get_cn_table.cache_clear()
    _get_cn_lookup.cache_clear()  # type: ignore
    _get_cn_polygons.cache_clear()
    _get_cn_border.cache_clear()
    _get_cn_line_strings.cache_clear()
    _get_cn_dataframe.cache_clear()
    _get_cn_geodataframe.cache_clear()
    get_countries.cache_clear()
    get_land.cache_clear()
    get_ocean.cache_clear()


@deprecator(alternative=get_cn_line)
def get_nine_line() -> shapely.MultiLineString:
    return cast(shapely.MultiLineString, get_cn_line())
