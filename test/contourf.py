import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib import patheffects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import frykit.plot as fplt

# 读取数据.
ds = xr.load_dataset('../frykit/data/test.nc')
t2m = ds['t2m'].isel(time=0) - 273.15
t2m[:] = gaussian_filter(t2m.values, sigma=1)

# 设置地图范围和刻度.
extents_main = [74, 136, 14, 56]
extents_sub = [105, 122, 2, 25]
xticks = np.arange(50, 161, 10)
yticks = np.arange(0, 71, 10)

# 设置投影.
crs_map = ccrs.LambertConformal(
    central_longitude=105, standard_parallels=(25, 47)
)
crs_data = ccrs.PlateCarree()

# 设置刻度风格.
plt.rc('xtick.major', size=8, width=0.9)
plt.rc('ytick.major', size=8, width=0.9)
plt.rc('xtick', labelsize=8, top=True, labeltop=True)
plt.rc('ytick', labelsize=8, right=True, labelright=True)

# 准备主地图.
fig = plt.figure(figsize=(10, 6))
ax_main = fig.add_subplot(111, projection=crs_map)
fplt.set_extent_and_ticks(
    ax_main, extents=extents_main,
    xticks=xticks, yticks=yticks,
    grid=True, lw=0.5, ls='--', color='gray'
)
# 添加要素.
ax_main.add_feature(cfeature.LAND.with_scale('50m'), fc='floralwhite')
ax_main.add_feature(cfeature.OCEAN.with_scale('50m'), fc='skyblue')
fplt.add_cn_province(ax_main, lw=0.3)
fplt.add_nine_line(ax_main, lw=0.5)
# 绘制填色图.
levels = np.linspace(0, 32, 50)
cticks = np.linspace(0, 32, 9)
cf = ax_main.contourf(
    t2m.longitude, t2m.latitude, t2m, levels,
    cmap='turbo', extend='both', transform=crs_data
)
fplt.clip_by_cn_border(cf, fix=True)
# 绘制colorbar.
cbar = fig.colorbar(
    cf, ax=ax_main, orientation='horizontal',
    shrink=0.6, pad=0.1, aspect=30,
    ticks=cticks, extendfrac=0
)
cbar.ax.tick_params(length=4, labelsize=8)
# 添加指北针和比例尺.
path_effects = [
    patheffects.Stroke(linewidth=2, foreground='w'),
    patheffects.Normal()
]
text_kwargs = {'path_effects': path_effects}
fplt.add_north_arrow(ax_main, (0.95, 0.9), text_kwargs=text_kwargs)
fplt.add_map_scale(
    ax_main, (0.1, 0.1), length=1000, ticks=[0, 500, 1000],
    text_kwargs=text_kwargs
)
# 设置标题.
ax_main.set_title(
    '2m Temerparture (\u2103)', y=1.1,
    fontsize='large', weight='bold'
)

# 准备小地图.
ax_sub = fig.add_axes(ax_main.get_position(), projection=crs_map)
ax_sub.set_extent(extents_sub, crs=crs_data)
ax_sub.gridlines(
    crs_data, xlocs=xticks, ylocs=yticks,
    lw=0.5, ls='--', color='gray'
)
ax_sub.add_feature(cfeature.LAND.with_scale('50m'), fc='floralwhite')
ax_sub.add_feature(cfeature.OCEAN.with_scale('50m'), fc='skyblue')
fplt.add_nine_line(ax_sub, lw=0.5)
# 只画出部分省份节省时间.
for name in ['广西壮族自治区', '广东省', '福建省', '海南省', '台湾省']:
    fplt.add_cn_province(ax_sub, name, lw=0.5)
cf = ax_sub.contourf(
    t2m.longitude, t2m.latitude, t2m, levels,
    cmap='turbo', extend='both', transform=crs_data
)
fplt.clip_by_cn_border(cf, fix=True)
fplt.add_map_scale(
    ax_sub, (0.5, 0.15), length=500,
    text_kwargs={'path_effects': path_effects}
)
# 最后定位ax_sub的位置.
fplt.locate_sub_axes(ax_main, ax_sub)

# 保存图片.
fig.savefig('../image/contourf.png', dpi=300, bbox_inches='tight')
plt.close(fig)