import matplotlib.pyplot as plt
import frykit.plot as fplt

# Axes的比例应该为1:1
fig, ax = plt.subplots()
ax.set_aspect('equal')

# 添加PathCollection后需要手动触发autoscale.
fplt.add_cn_province(ax, lw=0.5, fc='lightgray')
fplt.add_nine_line(ax, lw=0.5)
ax.autoscale_view()
ax.grid(ls='--')

# 手动添加刻度标签的度数符号.
degree = '\N{DEGREE SIGN}'
ax.xaxis.set_major_formatter('{x:.0f}' + degree + 'E')
ax.yaxis.set_major_formatter('{x:.0f}' + degree + 'N')

# 保存图片.
ax.set_title('Use Matplotlib Axes')
fig.savefig('../image/axes.png', dpi=300, bbox_inches='tight')
plt.close(fig)