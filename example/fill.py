import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import Normal, Stroke

import frykit.plot as fplt
import frykit.shp as fshp

# 虚构数据
provinces = fshp.get_cn_province()
data = np.linspace(0, 100, len(provinces))

# 设置地图范围
extents1 = [78, 134, 14, 55]
extents2 = [105, 120, 2, 25]

# 设置投影
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = ccrs.PlateCarree()

# 准备主地图
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(projection=map_crs)
ax1.set_extent(extents1, crs=data_crs)
fplt.add_nine_line(ax1, lw=0.5)
ax1.axis('off')

# 准备小地图
ax2 = fplt.add_mini_axes(ax1)
ax2.set_extent(extents2, crs=data_crs)
fplt.add_nine_line(ax2, lw=0.5)

# 填色参数
bins = [0, 20, 40, 60, 80, 100]
colors = ['#dbdee7', '#afc6e8', '#7a9cdc', '#3b6cb8', '#2a3b97']
nbin = len(bins) - 1
labels = [f'{bins[i]} - {bins[i + 1]}' for i in range(nbin)]

# 准备 colormap 和 norm
norm = mcolors.BoundaryNorm(bins, nbin)
cmap = mcolors.ListedColormap(colors)

# 字体描边
path_effects = [Stroke(linewidth=1.5, foreground='w'), Normal()]

# 绘制填色多边形，标注省名。
for ax in [ax1, ax2]:
    fplt.add_polygons(
        ax, provinces, array=data, cmap=cmap, norm=norm, ec='k', lw=0.4
    )
    for text in fplt.label_cn_province(ax):
        text.set_path_effects(path_effects)
        if text.get_text() in ['香港', '澳门']:
            text.set_visible(False)

# 添加图例
patches = []
for color, label in zip(colors, labels):
    patch = mpatches.Patch(fc=color, ec='k', lw=0.5, label=label)
    patches.append(patch)
ax1.legend(
    handles=patches,
    loc=(0.05, 0.05),
    frameon=False,
    handleheight=1.5,
    fontsize='small',
    title='data (units)',
)

# 添加指北针和比例尺
fplt.add_compass(ax1, 0.5, 0.85, size=20, style='circle')
scale_bar = fplt.add_scale_bar(ax1, 0.22, 0.1, length=1000)
scale_bar.set_xticks([0, 500, 1000])
scale_bar.set_xticks([250, 750], minor=True)

# 保存图片
fig.savefig('../image/fill.png', dpi=300, bbox_inches='tight')
plt.close(fig)
