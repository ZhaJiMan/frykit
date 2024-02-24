import numpy as np
import matplotlib.pyplot as plt
import frykit.plot as fplt

# Axes的比例应该为1:1
fig, ax = plt.subplots()
ax.set_aspect('equal')

# 添加要素和修饰.
fplt.add_ocean(ax, fc='#c4e7fa')
fplt.add_countries(ax, fc='#e7e4e2')
fplt.add_cn_province(ax, fc='#fcfeff')
fplt.add_nine_line(ax)
fplt.set_extent_and_ticks(
    ax=ax,
    extents=[70, 140, 0, 60],
    xticks=np.arange(-180, 181, 10),
    yticks=np.arange(-90, 91, 10),
)
fplt.gmt_style_frame(ax)
ax.grid(ls='--', c='gray')

# 保存图片.
ax.set_title('Use Matplotlib Axes', pad=15)
fig.savefig('../image/axes.png', dpi=300, bbox_inches='tight')
plt.close(fig)
