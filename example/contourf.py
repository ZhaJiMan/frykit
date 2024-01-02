import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import LAND
import frykit.plot as fplt

# 读取数据.
ds = fplt.load_test_nc()
X, Y = np.meshgrid(ds['longitude'], ds['latitude'])
Z = gaussian_filter(ds['t2m'] - 273.15, sigma=1)

# 设置地图范围和刻度.
extents1 = [74, 136, 13, 57]
extents2 = [105, 122, 2, 25]
xticks = np.arange(-180, 181, 10)
yticks = np.arange(-90, 91, 10)

# 设置投影.
map_crs = ccrs.AzimuthalEquidistant(
    central_longitude=105,
    central_latitude=35
)
data_crs = ccrs.PlateCarree()

# 设置刻度风格.
plt.rc('xtick.major', size=8, width=0.9)
plt.rc('ytick.major', size=8, width=0.9)
plt.rc('xtick', labelsize=8, top=True, labeltop=True)
plt.rc('ytick', labelsize=8, right=True, labelright=True)

# 准备主地图.
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(projection=map_crs)
fplt.set_extent_and_ticks(
    ax=ax1,
    extents=extents1,
    xticks=xticks,
    yticks=yticks
)
ax1.gridlines(
    xlocs=xticks,
    ylocs=yticks,
    lw=0.5,
    ls='--',
    color='gray'
)

# 添加要素.
ax1.set_facecolor('skyblue')  # 该投影中OCEAN的变换会出错.
ax1.add_feature(LAND.with_scale('50m'), fc='floralwhite')
fplt.add_cn_province(ax1, lw=0.3)
fplt.add_nine_line(ax1, lw=0.5)

# 绘制填色图.
levels = np.linspace(0, 32, 50)
cticks = np.linspace(0, 32, 9)
cf = ax1.contourf(
    X, Y, Z, levels,
    cmap='turbo',
    extend='both',
    transform=data_crs,
    transform_first=True
)
fplt.clip_by_cn_border(cf)

# 绘制colorbar.
cbar = fig.colorbar(
    cf,
    ax=ax1,
    orientation='horizontal',
    shrink=0.6,
    pad=0.1,
    aspect=30,
    ticks=cticks,
    extendfrac=0
)
cbar.ax.tick_params(length=4, labelsize=8)

# 添加指北针和比例尺.
fplt.add_compass(ax1, 0.92, 0.85, size=15, style='star')
map_scale = fplt.add_map_scale(ax1, 0.05, 0.1, length=1000)
map_scale.set_xticks([0, 500, 1000])
map_scale.xaxis.get_label().set_fontsize('small')

# 设置标题.
ax1.set_title(
    '2m Temerparture (\N{DEGREE CELSIUS})',
    y=1.1,
    fontsize='large',
    weight='bold'
)

# 准备小地图.
ax2 = fig.add_subplot(projection=map_crs)
ax2.set_extent(extents2, crs=data_crs)
fplt.move_axes_to_corner(ax2, ax1)
ax2.gridlines(
    xlocs=xticks,
    ylocs=yticks,
    lw=0.5,
    ls='--',
    color='gray'
)

# 添加要素.
ax2.set_facecolor('skyblue')
ax2.add_feature(LAND.with_scale('50m'), fc='floralwhite')
fplt.add_nine_line(ax2, lw=0.5)
fplt.add_cn_province(ax2, lw=0.3)
map_scale = fplt.add_map_scale(ax2, 0.4, 0.15, length=500)
map_scale.set_xticks([0, 500])
map_scale.xaxis.get_label().set_fontsize('small')

# 绘制填色图.
cf = ax2.contourf(
    X, Y, Z, levels,
    cmap='turbo',
    extend='both',
    transform=data_crs,
    transform_first=True
)
fplt.clip_by_cn_border(cf)

# 保存图片.
fig.savefig('../image/contourf.png', dpi=300, bbox_inches='tight')
plt.close(fig)