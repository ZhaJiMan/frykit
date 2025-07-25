from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum
from functools import cache
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict, cast, overload

import numpy as np
import pandas as pd
import shapely

from frykit import get_data_dir
from frykit.conf import DataSource, config
from frykit.shp.binary import BinaryReader
from frykit.shp.typing import LineStringType, PolygonType
from frykit.utils import deprecator, format_type_error, to_list

__all__ = [
    "get_cn_province_table",
    "get_cn_city_table",
    "get_cn_district_table",
    "get_cn_province_properties",
    "get_cn_city_properties",
    "get_cn_district_properties",
    "get_cn_province_names",
    "get_cn_city_names",
    "get_cn_district_names",
    "get_cn_province",
    "get_cn_city",
    "get_cn_district",
    "get_cn_border",
    "get_cn_line",
    "get_countries",
    "get_land",
    "get_ocean",
    "clear_data_cache",
    "get_nine_line",
    "clear_data_cache",
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
    file_path = _get_china_dir() / data_source / f"cn_{level}.csv"
    return pd.read_csv(file_path)


def _get_cn_province_table(data_source: DataSource | None) -> pd.DataFrame:
    data_source = _resolve_data_source(data_source)
    return _get_cn_table("province", data_source)


def _get_cn_city_table(data_source: DataSource | None) -> pd.DataFrame:
    data_source = _resolve_data_source(data_source)
    return _get_cn_table("city", data_source)


def _get_cn_district_table(data_source: DataSource | None) -> pd.DataFrame:
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


def _get_index_locs(index: pd.Index, key: Any) -> list[int]:
    """保证返回 list[int] 类型的 Index.get_loc"""
    loc = index.get_loc(key)
    match loc:
        case slice():
            return list(range(len(index))[loc])
        case np.ndarray():
            return np.nonzero(loc)[0].tolist()
        case _:
            return [loc]


NameOrAdcode: TypeAlias = int | str


def _get_cn_locs(
    names: pd.Index, adcodes: pd.Index, key: NameOrAdcode | Iterable[NameOrAdcode]
) -> list[int]:
    locs = []
    for k in to_list(key):
        match k:
            case str():
                locs.extend(_get_index_locs(names, k))
            case int():
                locs.extend(_get_index_locs(adcodes, k))
            case _:
                raise TypeError(format_type_error("key", k, [str, int]))

    return locs


def _get_cn_province_locs(
    province: NameOrAdcode | Iterable[NameOrAdcode] | None,
    data_source: DataSource | None,
) -> list[int]:
    df = _get_cn_province_table(data_source)
    if province is None:
        return list(range(len(df)))

    names = pd.Index(df["province_name"])
    adcodes = pd.Index(df["province_adcode"])
    locs = _get_cn_locs(names, adcodes, province)

    return locs


def _get_cn_city_locs(
    city: NameOrAdcode | Iterable[NameOrAdcode] | None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None,
    data_source: DataSource | None,
) -> list[int]:
    df = _get_cn_city_table(data_source)
    if city is None and province is None:
        return list(range(len(df)))
    if city is not None and province is not None:
        raise ValueError("不能同时指定 city 和 province")

    if city is not None:
        names = pd.Index(df["city_name"])
        adcodes = pd.Index(df["city_adcode"])
        key = city

    if province is not None:
        names = pd.Index(df["province_name"])
        adcodes = pd.Index(df["province_adcode"])
        key = province

    locs = _get_cn_locs(names, adcodes, key)  # type: ignore

    return locs


def _get_cn_district_locs(
    district: NameOrAdcode | Iterable[NameOrAdcode] | None,
    city: NameOrAdcode | Iterable[NameOrAdcode] | None,
    province: NameOrAdcode | Iterable[NameOrAdcode] | None,
    data_source: DataSource | None,
) -> list[int]:
    df = _get_cn_district_table(data_source)
    num_keys = sum(key is not None for key in [district, city, province])
    if num_keys == 0:
        return list(range(len(df)))
    if num_keys >= 2:
        raise ValueError("district、city 和 province 三个参数中只能指定一个")

    if district is not None:
        names = pd.Index(df["district_name"])
        adcodes = pd.Index(df["district_adcode"])
        key = district

    if city is not None:
        names = pd.Index(df["city_name"])
        adcodes = pd.Index(df["city_adcode"])
        key = city

    if province is not None:
        names = pd.Index(df["province_name"])
        adcodes = pd.Index(df["province_adcode"])
        key = province

    locs = _get_cn_locs(names, adcodes, key)  # type: ignore
    if isinstance(district, str) and len(locs) > 1:
        df_str = df.iloc[locs, :6].to_string(index=False)
        raise ValueError(f"存在复数个同名的县，请用 adcode 指定\n{df_str}")

    return locs


class ProvinceProperties(TypedDict):
    province_name: str
    province_adcode: int
    short_name: str
    lon: float
    lat: float


class CityProperties(TypedDict):
    province_name: str
    province_adcode: int
    city_name: str
    city_adcode: int
    short_name: str
    lon: float
    lat: float


class DistrictProperties(TypedDict):
    province_name: str
    province_adcode: int
    city_name: str
    city_adcode: int
    district_name: str
    district_adcode: int
    short_name: str
    lon: float
    lat: float


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
    locs = _get_cn_province_locs(province, data_source)
    result = df.iloc[locs].to_dict(orient="records")
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
    locs = _get_cn_city_locs(city, province, data_source)
    result = df.iloc[locs].to_dict(orient="records")
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
    locs = _get_cn_district_locs(district, city, province, data_source)
    result = df.iloc[locs].to_dict(orient="records")
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


@cache  # TODO: 不要加载所有数据
def _get_cn_polygons(level: AdminLevel, data_source: DataSource) -> list[PolygonType]:
    file_path = _get_china_dir() / data_source / f"cn_{level}.bin"
    with BinaryReader(file_path) as reader:
        polygons = reader.geometries()
        return cast(list[PolygonType], polygons)


def _get_cn_province_polygons(data_source: DataSource | None) -> list[PolygonType]:
    data_source = _resolve_data_source(data_source)
    return _get_cn_polygons("province", data_source)


def _get_cn_city_polygons(data_source: DataSource | None) -> list[PolygonType]:
    data_source = _resolve_data_source(data_source)
    return _get_cn_polygons("city", data_source)


def _get_cn_district_polygons(data_source: DataSource | None) -> list[PolygonType]:
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
    locs = _get_cn_province_locs(province, data_source)
    polygons = _get_cn_province_polygons(data_source)
    polygons = [polygons[loc] for loc in locs]

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
    locs = _get_cn_city_locs(city, province, data_source)
    polygons = _get_cn_city_polygons(data_source)
    polygons = [polygons[loc] for loc in locs]

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
    locs = _get_cn_district_locs(district, city, province, data_source)
    polygons = _get_cn_district_polygons(data_source)
    polygons = [polygons[loc] for loc in locs]

    if isinstance(district, (str, int)):
        return polygons[0]
    else:
        return polygons


@cache
def _get_cn_border(data_source: DataSource) -> shapely.MultiPolygon:
    file_path = _get_china_dir() / data_source / "cn_border.bin"
    with BinaryReader(file_path) as reader:
        return cast(shapely.MultiPolygon, reader.geometry(0))


def get_cn_border(data_source: DataSource | None = None) -> shapely.MultiPolygon:
    """获取中国国界的多边形"""
    data_source = _resolve_data_source(data_source)
    return _get_cn_border(data_source)


@cache
def _get_cn_line_strings() -> list[LineStringType]:
    file_path = _get_china_dir() / "cn_line.bin"
    with BinaryReader(file_path) as reader:
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
    file_path = _get_world_dir() / "country.bin"
    with BinaryReader(file_path) as reader:
        return cast(list[PolygonType], reader.geometries())


@cache
def get_land() -> shapely.MultiPolygon:
    """获取陆地多边形"""
    file_path = _get_world_dir() / "land.bin"
    with BinaryReader(file_path) as reader:
        return cast(shapely.MultiPolygon, reader.geometry(0))


@cache
def get_ocean() -> shapely.MultiPolygon:
    """获取海洋多边形"""
    file_path = _get_world_dir() / "ocean.bin"
    with BinaryReader(file_path) as reader:
        return cast(shapely.MultiPolygon, reader.geometry(0))


def clear_data_cache() -> None:
    """清除数据缓存"""
    _get_cn_table.cache_clear()
    _get_cn_polygons.cache_clear()
    _get_cn_border.cache_clear()
    _get_cn_line_strings.cache_clear()
    get_countries.cache_clear()
    get_land.cache_clear()
    get_ocean.cache_clear()


@deprecator(alternative=get_cn_line)
def get_nine_line() -> shapely.MultiLineString:
    return cast(shapely.MultiLineString, get_cn_line())
