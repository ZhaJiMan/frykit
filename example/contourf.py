import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.feature import LAND
from scipy.ndimage import gaussian_filter

import frykit.plot as fplt

# 读取数据
data = fplt.load_test_data()
X, Y = np.meshgrid(data['longitude'], data['latitude'])
Z = gaussian_filter(data['t2m'] - 273.15, sigma=1)

# 设置地图范围和刻度
extents1 = [74, 136, 13, 57]
extents2 = [105, 122, 2, 25]
xticks = np.arange(-180, 181, 10)
yticks = np.arange(-90, 91, 10)

# 设置投影
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = ccrs.PlateCarree()

# 准备主地图
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(projection=map_crs)
fplt.set_map_ticks(ax1, extents1, xticks, yticks)
ax1.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')

# 类似 NCL 的刻度风格
ax1.tick_params(
    length=8,
    width=0.9,
    labelsize=8,
    top=True,
    right=True,
    labeltop=True,
    labelright=True,
)

# 添加要素
land = LAND.with_scale('50m')
ax1.set_facecolor('skyblue')
ax1.add_feature(land, fc='floralwhite', ec='k', lw=0.5)
fplt.add_cn_province(ax1, lw=0.3)
fplt.add_nine_line(ax1, lw=0.5)

# 绘制填色图
levels = np.linspace(0, 32, 50)
cticks = np.linspace(0, 32, 9)
cf = ax1.contourf(
    X,
    Y,
    Z,
    levels,
    cmap='turbo',
    extend='both',
    transform=data_crs,
    transform_first=True,
)
fplt.clip_by_cn_border(cf)

# 绘制 colorbar
cbar = fig.colorbar(
    cf,
    ax=ax1,
    orientation='horizontal',
    shrink=0.6,
    pad=0.1,
    aspect=30,
    ticks=cticks,
    extendfrac=0,
)
cbar.ax.tick_params(length=4, labelsize=8)

# 添加指北针和比例尺
fplt.add_compass(ax1, 0.92, 0.85, size=15, style='star')
scale_bar = fplt.add_scale_bar(ax1, 0.05, 0.1, length=1000)
scale_bar.set_xticks([0, 500, 1000])
scale_bar.xaxis.get_label().set_fontsize('small')

# 设置标题
ax1.set_title(
    '2m Temerparture (\N{DEGREE CELSIUS})',
    y=1.1,
    fontsize='large',
    weight='bold',
)

# 准备小地图
ax2 = fplt.add_mini_axes(ax1)
ax2.set_extent(extents2, crs=data_crs)
ax2.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')

# 添加要素
ax2.set_facecolor('skyblue')
ax2.add_feature(land, fc='floralwhite', ec='k', lw=0.5)
fplt.add_cn_province(ax2, lw=0.3)
fplt.add_nine_line(ax2, lw=0.5)

# 绘制填色图
cf = ax2.contourf(
    X,
    Y,
    Z,
    levels,
    cmap='turbo',
    extend='both',
    transform=data_crs,
    transform_first=True,
)
fplt.clip_by_cn_border(cf)

# 添加比例尺
scale_bar = fplt.add_scale_bar(ax2, 0.4, 0.15, length=500)
scale_bar.set_xticks([0, 500])
scale_bar.xaxis.get_label().set_fontsize('small')

# 保存图片
fig.savefig('../image/contourf.png', dpi=300, bbox_inches='tight')
plt.close(fig)
