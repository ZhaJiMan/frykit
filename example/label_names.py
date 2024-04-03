import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import frykit.plot as fplt

# 设置投影
map_crs = ccrs.AzimuthalEquidistant(central_longitude=105, central_latitude=35)
data_crs = ccrs.PlateCarree()


def plot_province_map():
    '''绘制省界地图.'''
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection=map_crs)
    ax.set_extent([80, 126, 15, 54], crs=data_crs)
    fplt.add_cn_province(ax, lw=0.2, fc=plt.cm.Set3.colors)
    fplt.add_nine_line(ax, lw=0.5)
    fplt.label_cn_province(ax, fontsize='small')
    fig.savefig('../image/province_map.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_city_map():
    '''绘制市界地图.'''
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection=map_crs)
    ax.set_extent([80, 126, 15, 54], crs=data_crs)
    fplt.add_cn_city(ax, lw=0.2, fc=plt.cm.Set3.colors)
    fplt.add_nine_line(ax, lw=0.5)
    fplt.label_cn_city(ax, fontsize=5)
    fig.savefig('../image/city_map.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


plot_province_map()
plot_city_map()
