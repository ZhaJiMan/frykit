import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import frykit.plot as fplt

# 设置投影
map_crs = ccrs.AzimuthalEquidistant(
    central_longitude=105,
    central_latitude=35
)
data_crs = ccrs.PlateCarree()

# 绘制地图.
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection=map_crs)
ax.set_extent([80, 126, 15, 54], crs=data_crs)
fplt.add_cn_city(ax, lw=0.2, fc=plt.cm.Set3.colors)
fplt.add_nine_line(ax, lw=0.5)
fplt.label_cn_city(ax, fontsize=5)

# 保存图片.
fig.savefig('../image/city_map.png', dpi=300, bbox_inches='tight')
plt.close(fig)