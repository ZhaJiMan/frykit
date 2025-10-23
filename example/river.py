"""
绘制 LineString 类型的河流 shapefile

需要手动下载河流数据
"""

import matplotlib.pyplot as plt
from cartopy.io.shapereader import Reader

import frykit.plot as fplt

# 读取河流数据
# https://gaohr.win/site/blogs/2017/2017-04-18-GIS-basic-data-of-China.html
reader = Reader("R1/hyd1_4l.shp")
line_strings = list(reader.geometries())
reader.close()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection=fplt.CN_AZIMUTHAL_EQUIDISTANT)
fplt.set_map_ticks(ax, (78, 128, 15, 53))

ax.set_facecolor("#a4d7f6")
fplt.add_countries(ax, fc="#ffffff")
fplt.add_cn_province(ax, fc=plt.cm.Pastel2.colors)  # pyright: ignore[reportAttributeAccessIssue]
fplt.add_cn_line(ax)
fplt.add_geometries(ax, line_strings, fc="none", ec="#188ebf", zorder=2)

fplt.savefig("../image/river.png")
plt.close(fig)
