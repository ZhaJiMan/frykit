## `0.3.5`

- 移除 `make_nine_line_file` 函数。
- 移除 `get_dBZ_palette` 函数。
- 修改制作行政区划数据的脚本。
- 修改 `MANIFEST.in` 文件，降低安装包体积。

## `0.3.4`

- 以高德地图行政区域查询为数据源更新了 `data/shp` 目录里的矢量数据。
- `set_extent_and_ticks` 和 `add_map_scale` 函数现已支持普通 `Axes`。
- 改进 `get_cn_province` 和 `get_cn_city` 函数参数错误时的提示。

## `0.3.3`

- 新增 `_shp` 模块，改用自定义的二进制格式存储 shapefile 文件，减半数据文件体积的同时加快读取速度。
- 将 `get_cn_shp` 函数拆分为三个函数：
  - `get_cn_border`
  - `get_cn_province`
  - `get_cn_city`
- 新增模仿 GMT 边框风格的 `gmt_style_frame` 函数。
- 新增参考中央气象台雷达图配色的 `get_dBZ_palette` 函数。
- `add_polygons` 和 `add_cn_xxx` 系列函数现在能自动调整 `Axes` 的显示范围。
- 修正 `clip_by_polygon` 函数在 `matplotlib>=3.8.0` 时关于 `collections` 属性的问题。
- `add_compass` 函数新增 `circle` 样式。
- `plot_colormap` 函数新增 `extend` 参数。
- `make_qualitative_cmap` 函数改名为 `get_qualitative_palette`。

## `0.3.2`

- 新增 `image` 模块，提供拼图和切图的函数。
- 修正 `binned_statistic_2d` 函数颠倒 xy 的问题。

## `0.3.1`

- `clip_by_polygon` 函数可以接受一组 `Artist`。
- `add_box` 函数会对方框坐标做插值，保证方框在 `GeoAxes` 中平滑。
- 新增用于测试的 `load_test_nc` 函数。

## `0.3.0`

- 要求 `python>=3.9.0`。
- 所有代码加入类型提示。
- 修改 `interp_nearest` 函数对数组维度的要求。

## `0.2.5`

- `add_compass` 函数可以通过 `angle` 参数指定角度。

## `0.2.4`

- 修正 `add_map_scale` 函数里取线段的方式。
- 修正 `_transform` 函数里的 Shapely 版本问题。

## `0.2.3`

- 新增让 colormap 的白色对应于零值的 `CenteredBoundaryNorm` 类。
- 新增快速展示 colormap 和 normalize 的 `plot_colormap` 函数。

## `0.2.2`

- 要求 `cartopy>=0.20.0`。
- 修正 `set_extent_and_ticks` 函数受 `Axes.tick_params` 里 `top`、`right` 等参数影响的问题。

## `0.2.1`

- 修正 `clip_by_polygon` 函数在 `matplotlib<=3.6` 时报错的问题。

## `0.2.0`

- 以下函数加入缓存坐标变换结果的机制，只要维持对于多边形对象的引用，反复调用时就不会重复做坐标变换：
  - `add_polygon`
  - `add_polygons`
  - `clip_by_polygon`
- `add_cn_xxx` 系列函数会在模块内维持对中国行政区划数据的缓存，避免反复读取文件。
- 提供 `enable_fast_transform` 和 `disable_fast_transform` 函数切换坐标变换的方法。
- 增强 `clip_by_polygon` 和 `clip_by_polygons` 函数，能够避免直接使用 `set_clip_path` 时出界的问题。
- `add_polygon` 和 `add_polygons` 将 `crs=None` 解读为等经纬度投影。
- 统一使用 `set_extent_and_ticks` 函数设置 `GeoAxes` 的范围和刻度，并取消网格线相关的参数。
- `locate_sub_axes` 函数改名为 `move_axes_to_corner`。
- `add_north_arrow` 函数改名为 `add_compass`，并添加 `star` 样式。
- `get_cnshp` 函数改名为 `get_cn_shp`。