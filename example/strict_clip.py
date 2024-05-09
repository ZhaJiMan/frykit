import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import frykit.plot as fplt

# 准备扇形方框
extents = [100, 125, 15, 40]
lon0, lon1, lat0, lat1 = extents
path = fplt.rectangle_path(*extents).interpolated(100)

# 加载数据
data = fplt.load_test_data()
lon = data['longitude']
lat = data['latitude']
t2m = data['t2m']

# 设置地图
crs = ccrs.PlateCarree()
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10, 5),
    subplot_kw={'projection': fplt.CN_AZIMUTHAL_EQUIDISTANT},
)
for i, ax in enumerate(axes):
    fplt.add_cn_province(ax)
    ax.set_extent(extents, crs=crs)
    ax.set_boundary(path, transform=crs)
    ax.gridlines(draw_labels=True, rotate_labels=False, color='k', ls='--')

# 非严格裁剪会漏出来一点
pc1 = axes[0].pcolormesh(lon, lat, t2m, transform=crs)
fplt.clip_by_cn_border(pc1, strict=False)

# 严格裁剪
pc2 = axes[1].pcolormesh(lon, lat, t2m, transform=crs)
fplt.clip_by_cn_border(pc2, strict=True)

axes[0].set_title('strict=False', fontsize='large', color='r')
axes[1].set_title('strcit=True', fontsize='large', color='r')

plt.savefig('../image/strict_clip.png', dpi=300, bbox_inches='tight')
plt.close(fig)
