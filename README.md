# frykit

一个配合 Matplotlib 和 Cartopy 使用的工具箱，主要由 `shp` 和 `plot` 模块组成。

`shp` 模块的功能是：

- 读取中国行政区划数据
- 创建多边形掩膜（mask）
- 地理坐标变换

`plot` 模块的功能包括：

- 读取中国行政区划数据
- 向地图添加中国国界和省界
- 利用国界和省界对填色图进行裁剪（白化）
- 设置地图刻度
- 添加风矢量图的图例
- 添加指北针
- 添加地图比例尺

没有文档，但是每个函数都有详细的 docstring，可以在 Python命令行中通过 `help` 函数查看，或者在 IDE 中查看。

这个包只是作者自用的小工具集，函数编写粗糙，可能存在不少 bug，还请多多交流指正。类似的更完备的包还请移步 [gma](https://gma.luosgeo.com/) 或 [EOmaps](https://github.com/raphaelquast/EOmaps)。

## 安装

```
pip install frykit
```

依赖仅需 `cartopy>=0.20.0`。

## 示例

### 读取中国行政区划

```Python
import frykit.shp as fshp

# 读取中国国界.
country = fshp.get_cn_shp(level='国')

# 读取中国省界.
provinces = fshp.get_cn_shp(level='省')
anhui = fshp.get_cn_shp(level='省', province='安徽省')

# 读取中国市界.
cities = fshp.get_cn_shp(level='市')
hefei = fshp.get_cn_shp(level='市', city='合肥市')
cities_of_anhui = fshp.get_cn_shp(level='市', province='安徽省')
```

返回结果为 Shapely 的多边形对象，可以进行交并等几何运算。

行政区划的 shapefile 文件来自 [ChinaAdminDivisonSHP](https://github.com/GaryBikini/ChinaAdminDivisonSHP) 项目，坐标已从 GCJ-02 坐标系处理到了 WGS84 坐标系上。

### 绘制中国国界和省界

```Python
# 绘制中国国界.
fplt.add_cn_border(ax)

# 绘制中国省界.
fplt.add_cn_province(ax)
fplt.add_cn_province(ax, name=['安徽省', '江苏省'])
```

`ax` 可以为 `Axes` 或 `GeoAxes`。多次调用时能够通过缓存节省读取数据的时间开销。

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

只要在主程序中维持对 `polygon` 的引用，就能在多次调用时通过缓存节省坐标变换的时间开销。

Cartopy 的 `GeoAxes.add_geometries` 会自动去除 `polygons` 中不在 `GeoAxes` 显示范围内的多边形，破坏 `polygons` 和 `array` 的一一对应关系，打乱填色的结果。工具箱中的 `add_polygons` 函数不会进行这一操作，能够产生正确的填色结果。

### 裁剪填色图

```Python
cf = ax.contourf(lon, lat, data, transform=data_crs)
fplt.clip_by_cn_border(cf)
```

被裁剪的对象还可以是 `contour`、`clabel`、`pcolormesh`、`quiver` 等方法的返回值。

当用于裁剪的多边形超出 `GeoAxes` 的显示范围时，直接用 `Artist.set_clip_path` 做裁剪会发生填色图出界的现象（[cartopy/issues/2052](https://github.com/SciTools/cartopy/issues/2052)）。工具箱内的 `clip_by_xxx` 系列函数对此进行了处理。

### 添加指北针和比例尺

```Python
fplt.add_compass(ax1, 0.95, 0.8, size=15, style='star')
scale = fplt.add_map_scale(ax1, 0.36, 0.8, length=1000)
scale.set_xticks([0, 500, 1000])
```

指北针目前只是单纯指向图片上方，并不能真正指向所在地点的北方。

比例尺的尺寸是用地图中心处单位长度对应的距离计算得到。

### 定位南海小地图

```Python
sub_ax = fig.add_projection(projection=map_crs)
sub_ax.set_extent([105, 120, 2, 25], crs=data_crs)
fplt.move_axes_to_corner(sub_ax, ax)
```

需要先确定主图和子图的显示范围，再利用 `move_axes_to_corner` 函数将子图缩小并定位到主图的角落。

### 效果图

`example` 目录下有一些示例脚本：

![contourf](image/contourf.png)

![fill](image/fill.png)

![nerv_style](image/nerv_style.png)