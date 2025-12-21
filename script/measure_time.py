"""比较 cartopy 和 frykit 画中国地图的耗时"""

from __future__ import annotations

import timeit
from functools import partial
from io import BytesIO
from itertools import product
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.mpl.geoaxes import GeoAxes
from loguru import logger
from typing_extensions import assert_never

import frykit
import frykit.plot as fplt
import frykit.shp as fshp


def plot_map(
    small_map: bool = False,
    level: Literal["border", "province", "city", "district"] = "border",
    lib: Literal["cartopy", "frykit"] = "cartopy",
) -> None:
    if small_map:
        extents = (115, 120, 24, 28)
    else:
        extents = (70, 140, 0, 60)

    match level:
        case "border":
            data = fshp.get_cn_border()
        case "province":
            data = fshp.get_cn_province()
        case "city":
            data = fshp.get_cn_city()
        case "district":
            data = fshp.get_cn_district()
        case _:
            assert_never(level)

    match lib:
        case "cartopy":
            plot_func = lambda ax: ax.add_geometries(
                data, crs=fplt.PLATE_CARREE, fc="none", ec="k", lw=0.5
            )
        case "frykit":
            plot_func = lambda ax: fplt.add_geometries(
                ax, data, fc="none", ec="k", lw=0.5
            )
        case _:
            assert_never(lib)

    fig = plt.figure()
    ax = plt.axes(projection=fplt.CN_AZIMUTHAL_EQUIDISTANT)
    ax = cast(GeoAxes, ax)
    ax.set_extent(extents, crs=fplt.PLATE_CARREE)
    plot_func(ax)

    # 保存在内存中
    with BytesIO() as f:
        fplt.savefig(f, dpi=200)
    plt.close(fig)


def main() -> None:
    small_map_values = [False, True]
    levels = ("border", "province", "city", "district")
    libs = ("cartopy", "frykit")
    numbers = list(range(1, 4))

    index = pd.MultiIndex.from_product(
        [small_map_values, levels], names=["small_map", "level"]
    )
    columns = pd.MultiIndex.from_product([libs, numbers], names=["lib", "number"])

    for data_source in ("amap", "tianditu"):
        frykit.config.data_source = data_source
        df = pd.DataFrame(
            np.zeros((len(index), len(columns))), index=index, columns=columns
        )

        for small_map, level, lib in product(small_map_values, levels, libs):
            fshp.clear_data_cache()
            task = partial(plot_map, small_map, level, lib)
            times = timeit.repeat(task, number=1, repeat=len(numbers))
            for i, time in enumerate(times):
                df.loc[(small_map, level), (lib, i + 1)] = time
            logger.info(f"{data_source=}, {small_map=}, {level=}, {lib=}")

        df.round(2).to_csv(f"time_{data_source}.csv")


if __name__ == "__main__":
    main()
