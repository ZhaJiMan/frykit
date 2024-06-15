import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import frykit.plot as fplt

plt.rcParams['font.family'] = 'Microsoft YaHei'

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect(1)
fplt.set_map_ticks(ax, [105, 110.5, 28, 32.5], dx=1, dy=1)

# 绘制行政区划
fplt.add_cn_province(ax, fc='floralwhite')
fplt.add_cn_city(ax, '重庆城区', fc='springgreen', alpha=0.7)
fplt.add_cn_city(ax, '重庆郊县', fc='hotpink', alpha=0.7)
fplt.add_cn_district(ax, province='重庆市')
fplt.label_cn_district(ax, province='重庆市', fontsize='medium')

# 手动制作图例
patches = [
    Patch(color='springgreen', alpha=0.7, label='重庆城区'),
    Patch(color='hotpink', alpha=0.7, label='重庆郊县'),
]
ax.legend(
    handles=patches,
    loc='upper left',
    fontsize='large',
    framealpha=1,
    fancybox=False,
    edgecolor='k',
)
ax.set_title('重庆区县', fontsize='x-large')

fplt.savefig('../image/chongqing.png')
