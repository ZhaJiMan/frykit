"""比较高德和天地图数据在东北和台湾的差异"""

import matplotlib.pyplot as plt

import frykit
import frykit.plot as fplt

fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={"aspect": 1})

colors = plt.cm.Set3.colors  # type: ignore

for ax in axes[0, :]:
    fplt.set_map_ticks(ax, extents=(115, 130, 40, 55), dx=5, dy=5)
for ax in axes[1, :]:
    fplt.set_map_ticks(ax, extents=(118.5, 122.5, 21.5, 25.5), dx=1, dy=1)

for ax in axes[:, 0]:
    ax.set_title("amap", fontsize="x-large")
for ax in axes[:, 1]:
    ax.set_title("tianditu", fontsize="x-large")

# 临时设置数据源
with frykit.config.context(data_source="amap"):
    fplt.add_cn_province(axes[0, 0], fc=colors)
    fplt.label_cn_province(axes[0, 0])

    fplt.add_cn_district(axes[1, 0], fc=colors)
    fplt.label_cn_district(axes[1, 0])

with frykit.config.context(data_source="tianditu"):
    fplt.add_cn_province(axes[0, 1], fc=colors)
    fplt.label_cn_province(axes[0, 1])
    fplt.add_cn_district(axes[1, 1], fc=colors)
    fplt.label_cn_district(axes[1, 1])

fplt.savefig("../image/data_source.png")
plt.close(fig)
