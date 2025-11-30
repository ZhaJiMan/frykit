"""模仿 GeoDataFrame.plot，按数值给每个省份填色"""

from typing import cast

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.patheffects import Normal, Stroke

import frykit.plot as fplt
import frykit.shp as fshp

# 虚构数据
provinces = fshp.get_cn_province()
data = np.linspace(0, 100, len(provinces))

# 设置投影
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = ccrs.PlateCarree()

# 准备主地图
fig = plt.figure(figsize=(10, 6))
main_ax = fig.add_subplot(projection=map_crs)
main_ax = cast(GeoAxes, main_ax)
main_ax.set_extent((78, 134, 14, 55), data_crs)
fplt.add_cn_line(main_ax, lw=0.5)
main_ax.axis("off")

# 准备小地图
mini_ax = fplt.add_mini_axes(main_ax)
mini_ax = cast(GeoAxes, mini_ax)
mini_ax.set_extent((105, 120, 2, 25), data_crs)
fplt.add_cn_line(mini_ax, lw=0.5)

# 填色参数
bins = [0, 20, 40, 60, 80, 100]
colors = ["#dbdee7", "#afc6e8", "#7a9cdc", "#3b6cb8", "#2a3b97"]
nbin = len(bins) - 1
labels = [f"{bins[i]} - {bins[i + 1]}" for i in range(nbin)]

# 准备 cmap 和 norm
norm = BoundaryNorm(bins, nbin)
cmap = ListedColormap(colors)

# 字体描边
path_effects = [Stroke(linewidth=1.5, foreground="w"), Normal()]

# 绘制填色多边形，标注省名。
for ax in [main_ax, mini_ax]:
    fplt.add_geometries(ax, provinces, array=data, cmap=cmap, norm=norm, ec="k", lw=0.4)
    for text in fplt.label_cn_province(ax).texts:
        text.set_path_effects(path_effects)
        if text.get_text() in ["香港", "澳门"]:
            text.set_visible(False)

# 添加图例
patches = []
for color, label in zip(colors, labels):
    patch = Patch(fc=color, ec="k", lw=0.5, label=label)
    patches.append(patch)

main_ax.legend(
    handles=patches,
    loc=(0.05, 0.05),
    frameon=False,
    handleheight=1.5,
    fontsize="small",
    title="data (units)",
)

# 添加指北针和比例尺
fplt.add_compass(main_ax, 0.5, 0.85, size=20, style="circle")
scale_bar = fplt.add_scale_bar(main_ax, 0.22, 0.1, length=1000)
scale_bar.set_xticks([0, 500, 1000])
scale_bar.set_xticks([250, 750], minor=True)

# 保存图片
fplt.savefig("fill.png")
plt.close(fig)
