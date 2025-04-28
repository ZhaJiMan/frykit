"""比较 cartopy 和 frykit 画中国地图的耗时"""

import timeit
from functools import partial
from io import BytesIO
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import frykit.plot as fplt
import frykit.shp as fshp


def plot_map(small_map=False, level="border", lib="cartopy"):
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
            raise ValueError

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
            raise ValueError

    fig = plt.figure()
    ax = plt.axes(projection=fplt.CN_AZIMUTHAL_EQUIDISTANT)
    ax.set_extent(extents, crs=fplt.PLATE_CARREE)
    plot_func(ax)

    # 保存在内存中
    with BytesIO() as f:
        fplt.savefig(f, dpi=200)
    plt.close(fig)


def main():
    small_map_values = [False, True]
    levels = ["border", "province", "city", "district"]
    libs = ["cartopy", "frykit"]
    numbers = list(range(1, 4))

    index = pd.MultiIndex.from_product(
        [small_map_values, levels], names=["small_map", "level"]
    )
    columns = pd.MultiIndex.from_product([libs, numbers], names=["lib", "number"])
    df = pd.DataFrame(
        np.zeros((len(index), len(columns))), index=index, columns=columns
    )

    for small_map, level, lib in product(small_map_values, levels, libs):
        fshp.clear_data_cache()
        task = partial(plot_map, small_map, level, lib)
        times = timeit.repeat(task, number=1, repeat=len(numbers))
        for i, time in enumerate(times):
            df.loc[(small_map, level), (lib, i + 1)] = time
        print(f"{small_map=}, {level=}, {lib=}")

    df.round(2).to_csv("time.csv")


if __name__ == "__main__":
    main()
