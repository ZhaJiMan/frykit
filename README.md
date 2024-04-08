# frykit

一个配合 Matplotlib 和 Cartopy 使用的工具箱，主要由 `shp` 和 `plot` 模块组成。

`shp` 模块的功能是：

- 读取中国行政区划数据
- 创建多边形掩膜（mask）
- 多边形在不同投影坐标系之间的变换

`plot` 模块的功能包括：

- 读取中国行政区划数据
- 向地图添加中国国界、省界和市界
- 利用国界和省界对填色图做裁剪（白化）
- 标注省名和市名
- 设置地图刻度
- 添加风矢量图的图例
- 添加指北针
- 添加地图比例尺
- 制作离散色表

暂无文档，但是每个函数都有详细的 docstring，可以在 Python 命令行中通过 `help` 函数查看，或者在 IDE 中查看。

这个包只是作者自用的小工具集，函数编写粗糙，可能存在不少 bug，还请多多交流指正。类似的更完备的包还请移步 [gma](https://gma.luosgeo.com/) 或 [EOmaps](https://github.com/raphaelquast/EOmaps)。

## 安装

```
pip install frykit
```

依赖为：

```
python>=3.9.0
cartopy>=0.20.0
pandas>=1.2.0
```

## 更新记录

[CHANGELOG.md](https://github.com/ZhaJiMan/frykit/blob/main/CHANGELOG.md)

## 示例

### 读取中国行政区划

```Python
import frykit.shp as fshp

# 读取国界.
border = fshp.get_cn_border()

# 读取省界.
provinces = fshp.get_cn_province()
anhui = fshp.get_cn_province('安徽省')

# 读取市界.
cities = fshp.get_cn_city()
hefei = fshp.get_cn_city('合肥市')
cities_of_anhui = fshp.get_cn_city(province='安徽省')
```

返回结果是 [Shapely](https://shapely.readthedocs.io/en/stable/manual.html) 的多边形对象，可以进行交并等几何运算。

行政区划源数据来自 [高德地图行政区域查询接口](https://lbs.amap.com/api/webservice/guide/api/district)，含国界、省界和市界三套数据，已从 GCJ-02 坐标系处理到了 WGS84 坐标系上。文件都在 `frykit.DATA_DIRPATH` 指向的目录里。制作方法见 [amap-shp](https://github.com/ZhaJiMan/amap-shp)。

### 绘制中国国界、省界和市界

```Python
# 绘制国界.
fplt.add_cn_border(ax)

# 绘制九段线
fplt.add_nine_line(ax)

# 绘制省界.
fplt.add_cn_province(ax)
fplt.add_cn_province(ax, ['安徽省', '江苏省'])

# 绘制市界
fplt.add_cn_city(ax)
fplt.add_cn_city(ax, ['石家庄市', '保定市'])
fplt.add_cn_city(ax, province='河南省')
```

`ax` 可以是 `Axes` 或 `GeoAxes`。

### 标注省名和市名

```Python
fplt.label_cn_province(ax)
fplt.label_cn_city(ax, fontsize='xx-small')
```

默认采用 `Normal` 字重的思源黑体。

### 绘制全球数据

```Python
fplt.add_land(ax)
fplt.add_ocean(ax)
fplt.add_countries(ax)
```

全球数据在跨越地图投影坐标系的边界时很容易产生问题，需要小心使用。

### 绘制任意多边形

```Python
import shapely.geometry as sgeom

# 绘制一个多边形.
polygon = sgeom.polygon(...)
fplt.add_polygon(ax, polygon)

# 绘制多个多边形并填色.
pc = fplt.add_polygons(ax, polygons, array=data, cmap=cmap, norm=norm)
cbar = fig.colorbar(pc, ax=ax)
```

Cartopy 的 `GeoAxes.add_geometries` 会自动去除不在 `GeoAxes` 显示范围内的 `polygons`，破坏 `polygons` 和 `array` 的一一对应关系，打乱填色的结果。工具箱中的 `add_polygons` 函数不会进行这一操作，能够产生正确的填色结果。

### 裁剪填色图

```Python
cf = ax.contourf(lon, lat, data, transform=data_crs)

# 用国界或省界裁剪.
fplt.clip_by_cn_border(cf)
fplt.clip_by_cn_province(cf, '河北省')

# 用陆地裁剪.
fplt.clip_by_land(cf)
```

被裁剪的对象还可以是 `contour`、`clabel`、`pcolormesh`、`quiver` 等方法的返回值。

当用于裁剪的多边形超出 `GeoAxes` 的显示范围时，直接用 `Artist.set_clip_path` 做裁剪会发生填色图出界的现象（[cartopy/issues/2052](https://github.com/SciTools/cartopy/issues/2052)）。工具箱内的 `clip_by_xxx` 系列函数对此进行了处理。

### 填色图掩膜

```Python
border = fshp.get_cn_border()
mask = fshp.polygon_to_mask(border, lon, lat)
data[~mask] = np.nan
ax.contourf(lon, lat, data)
```

### 加速绘制和裁剪

绘制多边形和裁剪填色图过程中需要对多边形进行坐标变换，工具箱默认直接使用 pyproj 进行变换，速度快但可能在某些投影的边界产生错误的结果。为此可以手动切换回更正确的 Cartopy 的变换：

```Python
fplt.use_fast_transform(True)
fplt.add_cn_city(ax)  # 耗时1.6s

# 相当于ax.add_geometries
fplt.use_fast_transform(False)
fplt.add_cn_city(ax)  # 耗时31.6s
```

`add_cn_xxx` 系列函数在多次调用时会通过缓存节省读取国界和省界数据的时间开销。如果能维持对多边形对象的引用，`add_polygon`、`add_polygons` 和 `clip_by_polygon` 函数在多次调用时会通过缓存节省多边形坐标变换的时间开销。

### 设置地图范围和刻度

```Python
fplt.set_map_ticks(ax, extents=[-180, 180, -90, 90], dx=60, dy=30)
fplt.set_map_ticks(ax, xticks=[90, 100, 110], yticks=[20, 30, 40])
```

用 `dx` 和 `dy` 参数指定刻度间隔，或者通过数组直接指定刻度位置。

`mx` 和 `my` 参数指定小刻度的数量。

### 添加指北针

```Python
fplt.add_compass(ax, 0.95, 0.8, size=15)
```

`ax` 是 `GeoAxes` 时指北针会自动指向所在位置处的北向，也可以通过 `angle` 参数手动指定角度。

### 添加比例尺

```Python
scale_bar = fplt.add_scale_bar(ax1, 0.36, 0.8, length=1000)
scale_bar.set_xticks([0, 500, 1000])
```

比例尺的长度通过取样 `GeoAxes` 中心处单位长度对应的地理距离得到。比例尺对象类似 `Axes`，可以用 `set_xticks` 等方法进一步修改样式。

### 添加小地图

```Python
mini_ax = fplt.add_mini_axes(ax)
mini_ax.set_extent([105, 120, 2, 25], crs=data_crs)
fplt.add_cn_province(mini_ax)
```

小地图默认使用大地图的投影，会自动定位到大地图的角落，无需手动反复调节。

### 添加风矢量图例

```Python
fplt.add_quiver_legend(Q, U=10, width=0.15, height=0.12)
```

在 `Axes` 的角落添加一个白色矩形背景的风矢量图例。通过 `patch_kwargs` 字典控制背景的样式，`key_kwargs` 字典控制风箭头的样式。

### 添加经纬度方框

```Python
fplt.add_box(ax, [lon0, lon1, lat0, lat1], transform=ccrs.PlateCarree())
```

当 `ax` 是 `GeoAxes` 时会对方框上的点插值，以保证方框在 `ax` 的坐标系里足够平滑。

### GMT 风格边框

```Python
fplt.add_frame(ax)
```

使用类似 [GMT](https://www.generic-mapping-tools.org/) 黑白相间格子的边框。目前仅支持 `Axes`、等经纬度或墨卡托投影的 `GeoAxes`。

### 离散 colorbar

```Python
# 一个颜色对应一个刻度的定性colorbar.
colors = [
    'orangered', 'orange', 'yellow',
    'limegreen', 'royalblue', 'darkviolet'
]
cmap, norm, ticks = fplt.make_qualitative_cmap(colors)
cbar = fplt.plot_colormap(cmap, norm)
cbar.set_ticks(ticks)
cbar.set_ticklabels(colors)

# 保证零值区间对应白色的离散colorbar.
import cmaps
boundaries = [-10, -5, -2, -1, 1, 2, 5, 10, 20, 50, 100]
norm = fplt.CenteredBoundaryNorm(boundaries)
cbar = fplt.plot_colormap(cmaps.BlueWhiteOrangeRed, norm)
cbar.set_ticks(boundaries)
```

![colorbar](image/colorbar.png)

### 详细介绍

工具箱的原理和使用场景可见下面几篇博文：

- [Cartopy 系列：探索 shapefile](https://zhajiman.github.io/post/cartopy_shapefile/)
- [Cartopy 系列：裁剪填色图出界问题](https://zhajiman.github.io/post/cartopy_clip_outside/)
- [CALIPSO L2 VFM 产品的读取和绘制（with Python）](https://zhajiman.github.io/post/calipso_vfm/)
- [Matplotlib 系列：colormap 的设置](https://zhajiman.github.io/post/matplotlib_colormap/)

### 示例效果

`cd` 到包的 `example` 目录里可以执行示例脚本：

- [在普通 `Axes` 上画地图](example/axes.py)

![axes](image/axes.png)

- [墨卡托投影](example/mercator.py)

![mercator](image/mercator.png)

- [分省填色](example/fill.py)

![fill](image/fill.png)

- [剪裁 `contourf` 和 `quiver`](example/quiver.py)

![quiver](image/quiver.png)

- [剪裁主图和南海小图的 `contourf`](example/contourf.py)

![contourf](image/contourf.png)

- [模仿 NERV 风格的地图](example/nerv_style.py)

![nerv_style](image/nerv_style.png)