import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

import frykit.plot as fplt

# 构造假数据
npts = 100
x0, x1, y0, y1 = [70, 140, 0, 60]
x = np.linspace(x0, x1, npts)
y = np.linspace(y0, y1, npts)
X, Y = np.meshgrid(x, y)
Z = (np.cos(X / (x1 - x0) * np.pi) + np.sin(np.radians(Y)) * 2) * 2

# 设置地图
crs = ccrs.PlateCarree()
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=fplt.CN_AZIMUTHAL_EQUIDISTANT)
ax.set_extent([78, 128, 15, 55], crs=crs)
fplt.add_cn_border(ax)
fplt.add_nine_line(ax)

# 其实可以用国界去裁剪 clabel 的结果
# 但是会损失很多标签，所以这里直接手动点击
levels = np.linspace(0, 5, 11)
cf = ax.contourf(X, Y, Z, levels, cmap='rainbow', extend=True, transform=crs)
cs = ax.contour(X, Y, Z, levels, colors='k', linewidths=1, transform=crs)
labels = ax.clabel(cs, levels, manual=True, fontsize='large')
fplt.clip_by_cn_border(cf)
fplt.clip_by_cn_border(cs)

plt.savefig('../image/clabel.png', dpi=300, bbox_inches='tight')
plt.close(fig)
