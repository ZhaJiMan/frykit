import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import frykit.plot as fplt

# 设置地图投影.
map_crs1 = ccrs.PlateCarree()
map_crs2 = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = ccrs.PlateCarree()

# 设置地图范围.
extents1 = [70, 140, 10, 60]
extents2 = [78, 128, 15, 55]

# 设置方块网格范围.
dlon = dlat = 10
lon0, lon1, lat0, lat1 = 60, 150, 0, 70
xlocs = np.arange(lon0, lon1 + 1, dlon)
ylocs = np.arange(lat0, lat1 + 1, dlat)

# 准备两张地图.
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(wspace=0.1)
ax1 = fig.add_subplot(121, projection=map_crs1)
ax2 = fig.add_subplot(122, projection=map_crs2)
ax1.set_extent(extents1, crs=data_crs)
ax2.set_extent(extents2, crs=data_crs)
fplt.add_cn_province(ax1, lw=0.3, fc='floralwhite')
fplt.add_cn_province(ax2, lw=0.3, fc='floralwhite')
fplt.add_nine_line(ax1, lw=0.5)
fplt.add_nine_line(ax2, lw=0.5)
ax1.set_title('PlateCarree')
ax2.set_title('AzimuthalEquidistant')

# 交错画出方块格子.
for i in range(len(xlocs - 1)):
    for j in range(len(ylocs) - 1):
        if i % 2 == 0 and j % 2 == 0 or i % 2 == 1 and j % 2 == 1:
            x0 = lon0 + i * dlon
            x1 = x0 + dlon
            y0 = lat0 + j * dlat
            y1 = y0 + dlat
            for ax in [ax1, ax2]:
                fplt.add_box(
                    ax=ax,
                    extents=[x0, x1, y0, y1],
                    fc='royalblue',
                    ec='k',
                    zorder=2,
                    alpha=0.25,
                    transform=data_crs,
                )

# 保存图片.
fig.savefig('../image/add_box.png', dpi=300, bbox_inches='tight')
plt.close(fig)
