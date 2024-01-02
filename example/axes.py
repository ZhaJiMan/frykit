import matplotlib.pyplot as plt
import frykit.plot as fplt

# Axes的比例应该为1:1
fig, ax = plt.subplots()
ax.set_aspect('equal')

# 添加要素和修饰.
fplt.add_cn_province(ax, lw=0.5, fc='lightgray')
fplt.add_nine_line(ax, lw=0.5)
fplt.gmt_style_frame(ax)
ax.grid(ls='--')

# 手动添加刻度标签的度数符号.
degree = '\N{DEGREE SIGN}'
ax.xaxis.set_major_formatter('{x:.0f}' + degree + 'E')
ax.yaxis.set_major_formatter('{x:.0f}' + degree + 'N')

# 保存图片.
ax.set_title('Use Matplotlib Axes', pad=15)
fig.savefig('../image/axes.png', dpi=200, bbox_inches='tight')
plt.close(fig)