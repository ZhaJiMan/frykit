"""当 GeoAxes 的边界不是矩形时，用 strict_clip 参数防止裁剪出界"""

import matplotlib.pyplot as plt

import frykit.plot as fplt

# 准备扇形方框
extents = (100, 125, 15, 40)
lon0, lon1, lat0, lat1 = extents
path = fplt.box_path(*extents).interpolated(100)

# 加载数据
data = fplt.load_test_data()

# 设置地图
crs = fplt.PLATE_CARREE
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10, 5),
    subplot_kw={"projection": fplt.CN_AZIMUTHAL_EQUIDISTANT},
)
for i, ax in enumerate(axes):
    fplt.add_cn_province(ax)
    ax.set_extent(extents, crs=crs)
    ax.set_boundary(path, transform=crs)
    ax.gridlines(draw_labels=True, rotate_labels=False, color="k", ls="--")

# 非严格裁剪会漏出来一点
pc1 = axes[0].pcolormesh(data.lon, data.lat, data.t2m, transform=crs)
fplt.clip_by_cn_border(pc1)

# 严格裁剪
pc2 = axes[1].pcolormesh(data.lon, data.lat, data.t2m, transform=crs)
fplt.clip_by_cn_border(pc2, strict_clip=True)

axes[0].set_title("strict_clip=False", fontsize="large", color="r")
axes[1].set_title("strcit_clip=True", fontsize="large", color="r")

fplt.savefig("strict_clip.png")
plt.close(fig)
