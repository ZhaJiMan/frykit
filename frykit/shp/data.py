from collections.abc import Iterable
from itertools import chain
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from frykit import SHP_DIRPATH
from frykit.shp.binary import BinaryReader
from frykit.shp.typing import PolygonType

"""
数据源

- 省市县: https://lbs.amap.com/api/webservice/guide/api/district
- 九段线: https://datav.aliyun.com/portal/school/atlas/area_selector
- 所有国家: http://meteothink.org/downloads/index.html
- 海陆: https://www.naturalearthdata.com/downloads/50m-physical-vectors/
"""


class PrDict(TypedDict):
    pr_name: str
    pr_adcode: int
    short_name: str
    lon: float
    lat: float
    geometry: PolygonType


class CtDict(TypedDict):
    ct_name: str
    ct_adcode: int


class DtDict(TypedDict):
    dt_name: str
    dt_adcode: int


AdmKey = str | int | Iterable[str | int]
PrResult = PolygonType | list[PolygonType] | PrDict | list[PrDict]
CtResult = PolygonType | list[PolygonType] | CtDict | list[CtDict]
DtResult = PolygonType | list[PolygonType] | DtDict | list[DtDict]

# 缓存多边形数据
_data_cache: dict[str, BaseGeometry | list[BaseGeometry]] = {}


def clear_data_cache() -> None:
    """清除缓存的 shp 数据"""
    _data_cache.clear()


def get_cn_border() -> shapely.MultiPolygon:
    """获取中国国界的多边形"""
    polygon = _data_cache.get("cn_border")
    if polygon is None:
        filepath = SHP_DIRPATH / "cn_border.bin"
        with BinaryReader(filepath, region="china") as reader:
            polygon = reader.shape(0)
        _data_cache["cn_border"] = polygon

    return polygon


def get_nine_line() -> shapely.MultiPolygon:
    """获取九段线的多边形"""
    polygon = _data_cache.get("nine_line")
    if polygon is None:
        filepath = SHP_DIRPATH / "nine_line.bin"
        with BinaryReader(filepath, region="china") as reader:
            polygon = reader.shape(0)
        _data_cache["nine_line"] = polygon

    return polygon


def get_cn_province_table() -> pd.DataFrame:
    """获取省界元数据的表格"""
    filepath = SHP_DIRPATH / "cn_province.csv"
    return pd.read_csv(str(filepath))


def get_cn_city_table() -> pd.DataFrame:
    """获取市界元数据的表格"""
    filepath = SHP_DIRPATH / "cn_city.csv"
    return pd.read_csv(str(filepath))


def get_cn_district_table() -> pd.DataFrame:
    """获取县界元数据的表格"""
    filepath = SHP_DIRPATH / "cn_district.csv"
    return pd.read_csv(str(filepath))


_PR_TABLE = None
_CT_TABLE = None
_DT_TABLE = None


def _get_pr_table() -> pd.DataFrame:
    """获取缓存的省界元数据的表格"""
    global _PR_TABLE
    if _PR_TABLE is None:
        _PR_TABLE = get_cn_province_table()

    return _PR_TABLE


def _get_ct_table() -> pd.DataFrame:
    """获取缓存的市界元数据的表格"""
    global _CT_TABLE
    if _CT_TABLE is None:
        _CT_TABLE = get_cn_city_table()

    return _CT_TABLE


def _get_dt_table() -> pd.DataFrame:
    """获取缓存的县界元数据的表格"""
    global _DT_TABLE
    if _DT_TABLE is None:
        _DT_TABLE = get_cn_district_table()

    return _DT_TABLE


def _get_locs(index: pd.Index, key: Any) -> list[int]:
    """保证返回整数下标列表的 Index.get_loc"""
    loc = index.get_loc(key)
    if isinstance(loc, slice):
        return list(range(len(index))[loc])
    elif isinstance(loc, np.ndarray):
        return np.nonzero(loc)[0].tolist()
    else:
        return [loc]


def _get_pr_locs(province: AdmKey | None = None) -> list[int]:
    """查询省界元数据表格的下标"""
    df = _get_pr_table()
    if province is None:
        return list(range(len(df)))

    names = pd.Index(df["pr_name"])
    adcodes = pd.Index(df["pr_adcode"])

    def func(key: AdmKey) -> list[int]:
        if isinstance(key, str):
            return _get_locs(names, key)
        elif isinstance(key, int):
            return _get_locs(adcodes, key)
        else:
            return list(chain(*map(func, key)))

    return func(province)


def _get_ct_locs(
    city: AdmKey | None = None, province: AdmKey | None = None
) -> list[int]:
    """查询市界元数据表格的下标"""
    df = _get_ct_table()
    if city is None and province is None:
        return list(range(len(df)))
    if city is not None and province is not None:
        raise ValueError("不能同时指定 city 和 province")

    if city is not None:
        names = pd.Index(df["ct_name"])
        adcodes = pd.Index(df["ct_adcode"])
        key = city

    if province is not None:
        names = pd.Index(df["pr_name"])
        adcodes = pd.Index(df["pr_adcode"])
        key = province

    def func(key: AdmKey) -> list[int]:
        if isinstance(key, str):
            return _get_locs(names, key)
        elif isinstance(key, int):
            return _get_locs(adcodes, key)
        else:
            return list(chain(*map(func, key)))

    return func(key)


def _get_dt_locs(
    district: AdmKey | None = None,
    city: AdmKey | None = None,
    province: AdmKey | None = None,
) -> list[int]:
    """查询县界元数据表格的下标"""
    df = _get_dt_table()
    num_keys = sum(key is not None for key in [district, city, province])
    if num_keys == 0:
        return list(range(len(df)))
    if num_keys >= 2:
        raise ValueError("district、city 和 province 三个参数中只能指定一个")

    if district is not None:
        names = pd.Index(df["dt_name"])
        adcodes = pd.Index(df["dt_adcode"])
        key = district

    if city is not None:
        names = pd.Index(df["ct_name"])
        adcodes = pd.Index(df["ct_adcode"])
        key = city

    if province is not None:
        names = pd.Index(df["pr_name"])
        adcodes = pd.Index(df["pr_adcode"])
        key = province

    def func(key: AdmKey) -> list[int]:
        if isinstance(key, str):
            return _get_locs(names, key)
        elif isinstance(key, int):
            return _get_locs(adcodes, key)
        else:
            return list(chain(*map(func, key)))

    locs = func(key)
    if isinstance(district, str) and len(locs) > 1:
        lines = []
        for row in df.iloc[locs].itertuples(index=False):
            parts = [
                f"province={row.pr_name}",
                f"city={row.ct_name}",
                f"district={row.dt_name}",
                f"adcode={row.dt_adcode}",
            ]
            line = ", ".join(parts)
            lines.append(line)

        lines = "\n".join(lines)
        msg = f"存在复数个同名的区县，请用 adcode 指定\n{lines}"
        raise ValueError(msg)

    return locs


def get_cn_province_names(short: bool = False) -> list[str]:
    """获取中国省名"""
    df = _get_pr_table()
    key = "short_name" if short else "pr_name"
    names = df[key].tolist()

    return names


def get_cn_city_names(province: AdmKey | None = None, short=False) -> list[str]:
    """获取中国市名。可以指定获取某个省的所有市名。"""
    df = _get_ct_table()
    locs = _get_ct_locs(province=province)
    key = "short_name" if short else "ct_name"
    names = df[key].iloc[locs].tolist()

    return names


def get_cn_district_names(
    city: AdmKey | None = None,
    province: AdmKey | None = None,
    short: bool = False,
) -> list[str]:
    """获取中国县名。可以指定获取某个市或某个省的所有县名。"""
    df = _get_dt_table()
    locs = _get_dt_locs(city=city, province=province)
    key = "short_name" if short else "dt_name"
    names = df[key].iloc[locs].tolist()

    return names


def get_cn_province(province: AdmKey | None = None, as_dict: bool = False) -> PrResult:
    """
    获取中国省界的多边形

    Parameters
    ----------
    province : AdmKey, optional
        省名或 adcode。可以是复数个省。
        默认为 None，表示获取所有省。

    as_dict : bool, optional
        是否返回带元数据的字典。默认为 False。

    Returns
    -------
    result : PrResult
        表示省界的多边形或字典
    """
    polygons = _data_cache.get("cn_province")
    if polygons is None:
        filepath = SHP_DIRPATH / "cn_province.bin"
        with BinaryReader(filepath, region="china") as reader:
            polygons = reader.shapes()
        _data_cache["cn_province"] = polygons

    locs = _get_pr_locs(province)
    polygons = [polygons[i] for i in locs]

    if as_dict:
        df = _get_pr_table().iloc[locs]
        result = df.to_dict(orient="records")
        for d, polygon in zip(result, polygons):
            d["geometry"] = polygon
    else:
        result = polygons

    if isinstance(province, (str, int)):
        return result[0]
    return result


def get_cn_city(
    city: AdmKey | None = None,
    province: AdmKey | None = None,
    as_dict: bool = False,
) -> CtResult:
    """
    获取中国市界的多边形

    Parameters
    ----------
    city : AdmKey, optional
        市名或 adcode。可以是复数个市。
        默认为 None，表示获取所有市。

    province : AdmKey, optional
        省名或 adcode，表示获取某个省的所有市。可以是复数个省。
        默认为 None，表示不指定省。

    as_dict : bool, optional
        是否返回带元数据的字典。默认为 False。

    Returns
    -------
    result : CtResult
        表示市界的多边形或字典
    """
    polygons = _data_cache.get("cn_city")
    if polygons is None:
        filepath = SHP_DIRPATH / "cn_city.bin"
        with BinaryReader(filepath, region="china") as reader:
            polygons = reader.shapes()
        _data_cache["cn_city"] = polygons

    locs = _get_ct_locs(city, province)
    polygons = [polygons[i] for i in locs]

    if as_dict:
        df = _get_ct_table().iloc[locs]
        result = df.to_dict(orient="records")
        for d, polygon in zip(result, polygons):
            d["geometry"] = polygon
    else:
        result = polygons

    if isinstance(city, (str, int)):
        return result[0]
    return result


def get_cn_district(
    district: AdmKey | None = None,
    city: AdmKey | None = None,
    province: AdmKey | None = None,
    as_dict: bool = False,
) -> DtResult:
    """
    获取中国县界的多边形

    Parameters
    ----------
    district : AdmKey, optional
        县名或 adcode。可以是复数个县。
        默认为 None，表示获取所有县。

    city : AdmKey, optional
        市名或 adcode，表示获取某个市的所有县。可以是复数个市。
        默认为 None，表示不指定市。

    province : AdmKey, optional
        省名或 adcode，表示获取某个省的所有县。可以是复数个省。
        默认为 None，表示不指定省。

    as_dict : bool, optional
        是否返回带元数据的字典。默认为 False。

    Returns
    -------
    result : DtResult
        表示县界的多边形或字典
    """
    polygons = _data_cache.get("cn_district")
    if polygons is None:
        filepath = SHP_DIRPATH / "cn_district.bin"
        with BinaryReader(filepath, region="china") as reader:
            polygons = reader.shapes()
        _data_cache["cn_district"] = polygons

    locs = _get_dt_locs(district, city, province)
    polygons = [polygons[i] for i in locs]

    if as_dict:
        df = _get_dt_table().iloc[locs]
        result = df.to_dict(orient="records")
        for d, polygon in zip(result, polygons):
            d["geometry"] = polygon
    else:
        result = polygons

    if isinstance(district, (str, int)):
        return result[0]
    return result


def get_countries() -> list[PolygonType]:
    """获取所有国家国界的多边形"""
    polygons = _data_cache.get("country")
    if polygons is None:
        filepath = SHP_DIRPATH / "country.bin"
        with BinaryReader(filepath, region="world") as reader:
            polygons = reader.shapes()
        _data_cache["country"] = polygons

    return polygons


def get_land() -> shapely.MultiPolygon:
    """获取陆地多边形"""
    polygon = _data_cache.get("land")
    if polygon is None:
        filepath = SHP_DIRPATH / "land.bin"
        with BinaryReader(filepath, region="world") as reader:
            polygon = reader.shape(0)
        _data_cache["land"] = polygon

    return polygon


def get_ocean() -> shapely.MultiPolygon:
    """获取海洋多边形"""
    polygon = _data_cache.get("ocean")
    if polygon is None:
        filepath = SHP_DIRPATH / "ocean.bin"
        with BinaryReader(filepath, region="world") as reader:
            polygon = reader.shape(0)
        _data_cache["ocean"] = polygon

    return polygon
