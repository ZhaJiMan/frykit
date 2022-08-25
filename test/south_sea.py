import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import frykit.plot as fplt
import cmaps

# 读取数据.
ds = xr.load_dataset('data.nc')
t2m = ds['t2m'].isel(time=0) - 273.15
t2m[:] = gaussian_filter(t2m.values, sigma=1)

# 设置刻度风格.
plt.rc('xtick.major', size=8, width=0.9)
plt.rc('ytick.major', size=8, width=0.9)
plt.rc('xtick.minor', size=8, width=0.4)
plt.rc('ytick.minor', size=8, width=0.4)
plt.rc('xtick', labelsize=8, top=True)
plt.rc('ytick', labelsize=8, right=True)

# 绘制主地图.
crs = ccrs.PlateCarree()
fig = plt.figure()
ax_main = fig.add_subplot(111, projection=crs)
fplt.set_extent_and_ticks(
    ax_main, extents=[70, 140, 15, 55],
    xticks=np.arange(70, 141, 10),
    yticks=np.arange(15, 56, 10),
    nx=1, ny=1
)
fplt.add_cn_province(ax_main, lw=0.5)
fplt.add_nine_line(ax_main, lw=0.5)
levels = np.linspace(0, 32, 9)
cf = ax_main.contourf(
    t2m.longitude, t2m.latitude, t2m, levels,
    cmap=cmaps.ncl_default, extend='both', transform=crs
)
fplt.clip_by_cn_border(cf, fix=True)
cbar = fig.colorbar(
    cf, ax=ax_main, orientation='horizontal',
    shrink=0.9, pad=0.15, aspect=30,
    extendrect=True, extendfrac='auto'
)
cbar.ax.tick_params(length=0, labelsize=8)

# 绘制小地图.
ax_sub = fig.add_axes(ax_main.get_position(), projection=crs)
ax_sub.set_extent([105, 125, 2, 25], crs=crs)
# 只画出部分省份节省时间.
for name in ['广西壮族自治区', '广东省', '福建省', '海南省', '台湾省']:
    fplt.add_cn_province(ax_sub, name, lw=0.5)
fplt.add_nine_line(ax_sub, lw=0.5)
cf = ax_sub.contourf(
    t2m.longitude, t2m.latitude, t2m, levels,
    cmap=cmaps.ncl_default, extend='both', transform=crs
)
fplt.clip_by_cn_border(cf, fix=True)
fplt.locate_sub_axes(ax_main, ax_sub)

# 保存图片.
fig.savefig('south_sea.png', dpi=300)
plt.close(fig)