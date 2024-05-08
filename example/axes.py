import matplotlib.pyplot as plt

import frykit.plot as fplt

# Axes 的比例应该为 1:1
fig, ax = plt.subplots()
ax.set_aspect(1)

# 添加要素和修饰
ax.set_facecolor('#c4e7fa')
fplt.add_countries(ax, fc='#e7e4e2')
fplt.add_cn_province(ax, fc='#fcfeff')
fplt.add_nine_line(ax)
fplt.set_map_ticks(ax, [70, 140, 0, 60], dx=10, dy=10)
fplt.add_frame(ax)
ax.tick_params(length=10)
ax.grid(ls='--', c='gray')

# 保存图片
ax.set_title('Use Matplotlib Axes', pad=15)
fig.savefig('../image/axes.png', dpi=300, bbox_inches='tight')
plt.close(fig)
