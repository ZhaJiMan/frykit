"""绘制省市县三个级别的地图并标注地名"""

import matplotlib.pyplot as plt

import frykit.plot as fplt

# 设置投影
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = fplt.PLATE_CARREE


def plot_province_map():
    """绘制省界地图"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection=map_crs)
    ax.set_extent((80, 126, 15, 54), crs=data_crs)
    fplt.add_cn_province(ax, lw=0.2, fc=plt.cm.Set3.colors)
    fplt.add_cn_line(ax, lw=0.5)
    fplt.label_cn_province(ax, fontsize="small")
    fplt.savefig("../image/province_map.png")
    plt.close(fig)


def plot_city_map():
    """绘制市界地图"""
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection=map_crs)
    ax.set_extent((80, 126, 15, 54), crs=data_crs)
    fplt.add_cn_city(ax, lw=0.2, fc=plt.cm.Set3.colors)
    fplt.add_cn_line(ax, lw=0.5)
    fplt.label_cn_city(ax, fontsize=5)
    fplt.savefig("../image/city_map.png")
    plt.close(fig)


def plot_district_map():
    """绘制县界地图"""
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection=map_crs)
    ax.set_extent((80, 126, 15, 54), crs=data_crs)
    fplt.add_cn_district(ax, lw=0.2, fc=plt.cm.Set3.colors)
    fplt.add_cn_line(ax, lw=0.5)
    # fplt.label_cn_district(ax, fontsize=5)
    fplt.savefig("../image/district_map.png")
    plt.close(fig)


plot_province_map()
plot_city_map()
plot_district_map()
