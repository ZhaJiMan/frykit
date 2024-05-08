# https://scitools.org.uk/cartopy/docs/latest/matplotlib/intro.html
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Geod

import frykit.plot as fplt

# 设置投影
data_crs = ccrs.PlateCarree()
map_crs = fplt.WEB_MERCATOR

# 设置起点和终点
lon1, lat1 = -60, 70
lon2, lat2 = 60, -70
x1, y1 = map_crs.transform_point(lon1, lat1, data_crs)
x2, y2 = map_crs.transform_point(lon2, lat2, data_crs)

npts = 100
lons = np.linspace(lon1, lon2, npts)
lats = np.linspace(lat1, lat2, npts)
xs = np.linspace(x1, x2, npts)
ys = np.linspace(y1, y2, npts)

# 经纬度连线和等角航线
plate_carree_line = np.c_[lons, lats]
rhumb_line = data_crs.transform_points(map_crs, xs, ys)[:, :2]

# 大圆航线
geod = Geod(ellps='WGS84')
r = geod.inv_intermediate(lon1, lat1, lon2, lat2, npts)
greate_circle_line = np.c_[r.lons, r.lats]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection=map_crs)
fplt.set_map_ticks(ax, dx=60, yticks=np.arange(-80, 81, 20))
ax.stock_img()

# 绘制三种路线
ax.plot(
    rhumb_line[:, 0],
    rhumb_line[:, 1],
    'k',
    transform=data_crs,
    label='Rhumb Line',
)
ax.plot(
    plate_carree_line[:, 0],
    plate_carree_line[:, 1],
    'r--',
    transform=data_crs,
    label='PlateCarre Line',
)
ax.plot(
    greate_circle_line[:, 0],
    greate_circle_line[:, 1],
    'b--',
    transform=data_crs,
    label='Great Circle Line',
)
ax.plot([lon1, lon2], [lat1, lat2], 'ko', transform=data_crs)

ax.legend(framealpha=1, fontsize='large')
ax.set_title('Different Lines in Web Mercator Map', fontsize='x-large')

# 保存图片
fig.savefig('../image/mercator.png', dpi=300, bbox_inches='tight')
plt.close(fig)
