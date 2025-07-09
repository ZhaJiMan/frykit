"""
模仿 EVA 中 NERV 风格的地图

需要安装 opencc 将省名繁体化，并安装 FOT-Matisse Pro 字体
"""

import matplotlib.pyplot as plt
import opencc

import frykit.plot as fplt
import frykit.shp as fshp

# 线条颜色
linecolor = "#a3ffc2"
boxcolor = "#f29305"
fontcolor = "#ffc292"


# 繁体化省名
converter = opencc.OpenCC("s2t.json")
df = fshp.get_cn_province_table()
names = df["short_name"].tolist()
for i, name in enumerate(names):
    if name == "香港" or name == "澳门":
        name = ""
    names[i] = converter.convert(name)

# 设置投影
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = fplt.PLATE_CARREE

# 创建画布
fig = plt.figure(facecolor="k")
ax = fig.add_subplot(111, projection=map_crs)
ax.set_extent((76, 134, 2, 55), crs=data_crs)
ax.set_facecolor("k")
ax.axis("off")

# 绘制省界和市界
fplt.add_cn_province(ax, lw=0.6, ec=linecolor)
fplt.add_cn_city(ax, lw=0.2, ec=linecolor)
fplt.add_cn_line(ax, lw=0.6, ec=linecolor)

# 两种风格的方框
hollow_props = {
    "boxstyle": "round, rounding_size=0.2",
    "fc": "none",
    "ec": boxcolor,
    "lw": 1,
    "alpha": 0.8,
}
solid_props = {
    "boxstyle": "round, rounding_size=0.2",
    "fc": boxcolor,
    "ec": "none",
    "alpha": 0.8,
}

# 添加说明框
ax.text(
    x=0.2,
    y=0.2,
    s="CHINA MAP",
    color=boxcolor,
    alpha=hollow_props["alpha"],
    fontsize=10,
    fontfamily="Helvetica Neue",
    fontweight="bold",
    ha="left",
    va="center",
    bbox=hollow_props,
    transform=ax.transAxes,
)
ax.text(
    x=0.2,
    y=0.26,
    s="SEP 2022",
    color=boxcolor,
    alpha=hollow_props["alpha"],
    fontsize=10,
    fontfamily="Helvetica Neue",
    fontweight="bold",
    ha="left",
    va="center",
    transform=ax.transAxes,
)

# 添加省名
for name, lon, lat in zip(names, df["lon"], df["lat"]):
    ax.text(
        x=lon,
        y=lat,
        s=name,
        color=fontcolor,
        fontsize=4,
        fontfamily="FOT-Matisse Pro",
        ha="center",
        va="center",
        bbox=solid_props,
        transform=data_crs,
    )

# 保存图片
fplt.savefig("../image/nerv_style.png")
plt.close(fig)
