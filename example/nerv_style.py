import opencc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import frykit.shp as fshp
import frykit.plot as fplt

# 线条颜色.
linecolor = '#a3ffc2'
boxcolor = '#f29305'
fontcolor = '#ffc292'

# 读取shp文件记录.
provinces = []
names = fshp.get_cn_province_names(short=True)
provinces = fshp.get_cn_shp(level='省')
cities = fshp.get_cn_shp(level='市')

# 繁体化省名.
converter = opencc.OpenCC('s2t.json')
for i, name in enumerate(names):
    if name == '香港' or name == '澳门':
        name = ''
    names[i] = converter.convert(name)

# 设置投影.
map_crs = ccrs.AzimuthalEquidistant(
    central_longitude=105,
    central_latitude=35
)
data_crs = ccrs.PlateCarree()

# 创建画布.
fig = plt.figure(facecolor='k')
ax = fig.add_subplot(111, projection=map_crs)
ax.set_extent([76, 134, 2, 55], crs=data_crs)
ax.set_facecolor('k')
ax.axis('off')

# 绘制省界和市界.
fplt.add_polygons(ax, provinces, lw=0.6, fc='none', ec=linecolor)
fplt.add_polygons(ax, cities, lw=0.2, fc='none', ec=linecolor)
fplt.add_nine_line(ax, lw=0.6, ec=linecolor)

# 两种风格的方框.
hollow_props = {
    'boxstyle': 'round, rounding_size=0.2',
    'fc': 'none', 'ec': boxcolor,
    'lw': 1, 'alpha': 0.8
}
solid_props = {
    'boxstyle': 'round, rounding_size=0.2',
    'fc': boxcolor, 'ec': 'none', 'alpha': 0.8
}

# 添加说明框.
ax.text(
    0.2, 0.2, 'CHINA MAP',
    color=boxcolor,
    alpha=hollow_props['alpha'],
    fontsize=10,
    fontfamily='Helvetica Neue',
    fontweight='bold',
    ha='left',
    va='center',
    bbox=hollow_props,
    transform=ax.transAxes
)
ax.text(
    0.2, 0.26, 'SEP 2022',
    color=boxcolor,
    alpha=hollow_props['alpha'],
    fontsize=10,
    fontfamily='Helvetica Neue',
    fontweight='bold',
    ha='left',
    va='center',
    transform=ax.transAxes
)

# 添加省名.
lonlats = fshp.get_cn_province_lonlats()
for name, (lon, lat) in zip(names, lonlats):
    ax.text(
        lon, lat, name,
        color=fontcolor,
        fontsize=4,
        fontfamily='FOT-Matisse Pro',
        ha='center',
        va='center',
        bbox=solid_props,
        transform=data_crs
    )

# 保存图片.
fig.savefig('../image/nerv_style.png', dpi=300, bbox_inches='tight')
plt.close(fig)