from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import frykit.plot as fplt
import frykit.shp as fshp

# 读取shp文件记录.
names = []
provinces = []
for record in fshp.get_cnshp(level='省', as_dict=True):
    names.append(record['pr_name'])
    provinces.append(record['geometry'])

# 生成0-100的随机数据.
rng = default_rng(1)
data = rng.random(len(names)) * 100

# 设置地图范围.
extents_main = [78, 134, 14, 55]
extents_sub = [105, 120, 2, 25]

# 设置投影.
crs_map = ccrs.LambertConformal(
    central_longitude=105, standard_parallels=(25, 47)
)
crs_data = ccrs.PlateCarree()

# 准备主地图.
fig = plt.figure(figsize=(10, 6))
ax_main = fig.add_subplot(111, projection=crs_map)
ax_main.set_extent(extents_main, crs=crs_data)
fplt.add_nine_line(ax_main, lw=0.5)
ax_main.axis('off')

# 准备小地图.
ax_sub = fig.add_axes(ax_main.get_position(), projection=crs_map)
ax_sub.set_extent(extents_sub, crs=crs_data)
fplt.add_nine_line(ax_sub, lw=0.5)
fplt.locate_sub_axes(ax_main, ax_sub, shrink=0.4)

# 填色参数.
bins = [0, 20, 40, 60, 80, 100]
colors = ['#dbdee7', '#afc6e8', '#7a9cdc', '#3b6cb8', '#2a3b97']
nbin = len(bins) - 1
labels = [f'{bins[i]} - {bins[i + 1]}' for i in range(nbin)]

# 准备colormap和norm.
norm = mcolors.BoundaryNorm(bins, nbin)
cmap = mcolors.ListedColormap(colors)
# 绘制填色的多边形
for ax in [ax_main, ax_sub]:
    fplt.add_polygons(
        ax, provinces, crs_data, array=data,
        cmap=cmap, norm=norm, ec='k', lw=0.4
    )

# 添加图例.
patches = []
for color, label in zip(colors, labels):
    patch = mpatches.Patch(fc=color, ec='k', lw=0.5, label=label)
    patches.append(patch)
ax_main.legend(
    handles=patches, loc=(0.05, 0.05),
    frameon=False, handleheight=1.5, fontsize='small',
    title='data (units)'
)

# 添加指北针和比例尺.
fplt.add_north_arrow(ax_main, (0.1, 0.85))
fplt.add_map_scale(ax_main, (0.45, 0.8), length=1000, ticks=[0, 500, 1000])

# 简化名称.
for i, name in enumerate(names):
    if '香港' in name or '澳门' in name:
        names[i] = ''
    elif '内蒙古' in name or '黑龙江' in name:
        names[i] = name[:3]
    else:
        names[i] = name[:2]

# 添加省名.
for name, province in zip(names, provinces):
    point = province.representative_point()
    ax_main.text(
        point.x, point.y, name,
        ha='center', va='center', fontsize='xx-small',
        fontfamily='Source Han Sans SC', transform=crs_data
    )

# 保存图片.
fig.savefig('../image/fill.png', dpi=300, bbox_inches='tight')
plt.close(fig)