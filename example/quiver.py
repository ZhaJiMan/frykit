import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import frykit.plot as fplt

# 读取数据.
data = fplt.load_test_data()
X, Y = np.meshgrid(data['longitude'], data['latitude'])
t2m = gaussian_filter(data['t2m'] - 273.15, sigma=1)
u10 = gaussian_filter(data['u10'], sigma=1)
v10 = gaussian_filter(data['v10'], sigma=1)

# 设置地图范围.
extents = [78, 128, 15, 55]

# 设置投影.
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = ccrs.PlateCarree()

# 准备地图.
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(projection=map_crs)
ax.set_extent(extents, crs=data_crs)
fplt.add_cn_province(ax, lw=0.4)

# 绘制气温.
cf = ax.contourf(
    X,
    Y,
    t2m,
    levels=np.linspace(-10, 35, 10),
    cmap=plt.cm.plasma,
    extend='both',
    transform=data_crs,
    transform_first=True,
)
fig.colorbar(cf, ax=ax, label='Temperature (℃)')

# 绘制风场.
Q = ax.quiver(
    X,
    Y,
    u10,
    v10,
    scale=0.15,
    scale_units='dots',
    regrid_shape=35,
    transform=data_crs,
)
fplt.add_quiver_legend(Q, U=10, height=0.12)

# 裁剪结果.
fplt.clip_by_cn_border(cf)
fplt.clip_by_cn_border(Q)

# 保存图片.
fig.savefig('../image/quiver.png', dpi=300, bbox_inches='tight')
plt.close(fig)
