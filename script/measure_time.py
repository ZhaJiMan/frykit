import os
from typing import Literal

import cartopy
import matplotlib.pyplot as plt
import pandas as pd

import frykit
import frykit.plot as fplt
import frykit.shp as fshp
from frykit.prof import timer

print(cartopy.__version__)
print(frykit.__version__)


def measure_time(n: int = 4, region=Literal['china', 'south']) -> None:
    if region == 'china':
        extents = (70, 140, 0, 60)
    if region == 'south':
        extents = (115, 117, 25, 27)

    result1 = {}
    result2 = {}

    for level in ['border', 'province', 'city', 'district']:
        match level:
            case 'border':
                polygons = fshp.get_cn_border()
            case 'province':
                polygons = fshp.get_cn_province()
            case 'city':
                polygons = fshp.get_cn_city()
            case 'district':
                polygons = fshp.get_cn_district()

        times1 = []
        times2 = []

        @timer(out=times1, verbose=True)
        def cartopy_func():
            crs = fplt.PLATE_CARREE
            fig = plt.figure()
            ax = fig.add_subplot(projection=fplt.CN_AZIMUTHAL_EQUIDISTANT)
            ax.set_extent(extents, crs=crs)
            ax.add_geometries(polygons, crs, fc='none', ec='k', lw=0.5)
            fplt.savefig('cartopy.png', dpi=200)
            plt.close(fig)

        @timer(out=times2, verbose=True)
        def frykit_func():
            crs = fplt.PLATE_CARREE
            fig = plt.figure()
            ax = fig.add_subplot(projection=fplt.CN_AZIMUTHAL_EQUIDISTANT)
            ax.set_extent(extents, crs=crs)
            fplt.add_geoms(ax, polygons, fc='none', ec='k', lw=0.5)
            fplt.savefig('frykit.png', dpi=200)
            plt.close(fig)

        for _ in range(n):
            cartopy_func()
            frykit_func()

        result1[level] = times1
        result2[level] = times2

    df1 = pd.DataFrame(result1).round(3)
    df1.index = df1.index + 1
    df1.to_csv(f'cartopy_{region}.csv')

    df2 = pd.DataFrame(result2).round(3)
    df2.index = df2.index + 1
    df2.to_csv(f'frykit_{region}.csv')

    os.remove('cartopy.png')
    os.remove('frykit.png')


if __name__ == '__main__':
    measure_time(region='china')
    # measure_time(region='south')
