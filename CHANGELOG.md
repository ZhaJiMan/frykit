## `0.5.3`

- 新增保存图片的 `savefig` 函数，相当于有默认参数的 `Figure.savefig`。
- `add_mini_axes` 函数新增 `aspect` 参数，用于修改 `Axes` 的宽高。
- 新增经纬度换算成球面 xyz 坐标的 `lon_lat_to_xyz` 函数。
- 移除 `add_polygon` 函数，现在 `add_polygons` 函数也能绘制单个多边形。
- 移除 `clip_by_polygons` 函数。
- 更新 README。

## `0.5.2`

- 新增快速创建中国地图的 `quick_cn_map` 函数。
- `add_side_axes` 函数的 `depth` 参数更名为 `width`。
- 修改 `add_texts` 函数的默认效果。
- 修改 `make_gif` 函数的参数。

## `0.5.1`

- 版本号被浪费了，遂跳过。

## `0.5.0`

- 新增 `_artist` 模块，将风矢量图例、指北针、比例尺和 GMT 边框改用 `Artist` 实现，保证它们能自动更新状态。
- `add_map_scale` 函数更名为 `add_scale_bar`。
- `gmt_style_frame` 函数更名为 `add_frame`，内部不再修改 `ax` 的刻度样式。
- 移除 `move_axes_to_corner` 函数，新增功能更强的 `add_mini_axes` 函数。
- 移除 `get_cn_shp` 函数。
- 修改 `get_cn_xxx` 系列函数的参数名，修改它们查询地名的方式，现在只要有一个地名出错就会报错。另外移除了它们的 `as_dict` 参数（因为没见有人用过……）。
- 改进 `calc` 模块里角度转换的函数输出的数值范围。
- `deprecator` 装饰器新增 `raise_error` 参数。
- 将 `loc` 参数的取值 `bottom left` 和 `bottom right` 修改为 `lower left` 和 `lower right`。
- 修正一些函数的类型提示。
- 所有代码用 isort 做了格式化。
- 添加 license。

## `0.4.5`

- 新增 `add_texts` 函数。
- `ntick` 参数更名为 `nticks`。

## `0.4.4`

- `set_extent_and_ticks` 函数更名为 `set_map_ticks` 函数，新增 `dx` 和 `dy` 参数，`nx` 和 `ny` 参数改名为 `mx` 和 `my`。非等经纬度投影现已支持开启次刻度。
- `load_test_nc` 函数更名为 `load_test_data`，移除对 xarray 和 netCDF4 的依赖。
- `image` 模块各函数的输入现在可以是 `Image` 对象。
- 新增墨卡托投影的例子。为避免错误，简化 `contourf.py` 例子。

## `0.4.3`

- 将影响坐标变换速度的 `Projection._as_mpl_transform` 修改成了 `Axes.transData`。
- 新增时分秒转换度数的 `hms_to_degrees2` 函数。
- 新增比较两张图片的 `compare_images` 函数。

## `0.4.2`

- 增加全球国家国界、50m 陆地和海洋的数据。相关函数：
  - `get_countries`、`add_countries`
  - `get_land`、`add_land`
  - `get_ocean`、`add_ocean`
- 为了加入全球数据，修改了 `_shp` 模块的压缩参数。
- 取消 `clip_by_land` 和 `clip_by_ocean` 的 `resolution` 参数。
- 移除 `enable_fast_transform` 和 `disable_fast_transform` 函数，用 `use_fast_transform` 函数替代。
- 新增 `GeometryTransformer` 类，提高了 `plot` 模块中对大量多边形做变换的速度。
- 修改 `cn_province.csv` 和 `cn_city.csv` 里标注用的经纬度。
- 修改 `example` 中的例子。
- 所有代码用 Black 做了格式化。

## `0.4.1`

- 明确安装依赖。

## `0.4.0`

- 通过 `mapshaper` 简化 shapefile 文件，安装包体积减小到 `0.3.0` 的三分之一。
- Shapefile 文件属性中加入省名市名的简写和用来标注名称的经纬度。
- 新增返回经纬度的 `get_cn_province_lonlats` 和 `get_cn_city_lonlats()` 函数。
- 新增绘制中国市界的 `add_cn_city` 函数。
- 新增标注省名和市名的 `label_cn_province` 和 `label_cn_city` 函数。
- `get_cn_province_names` 和 `get_cn_city_names` 函数新增 `short` 参数。
- 修改 `get_cn_province` 和 `get_cn_city` 是否返回列表的逻辑。
- `clip_by_polygon` 函数现在默认使用 `set_clip_box` 防止裁剪出界，新增 `strict` 参数决定是否进行更严格的裁剪。
- 新增 x 轴夹角和方位角互相转换的 `t_to_az` 和 `az_to_t` 函数。
- `xy_to_polar` 和 `polar_to_xy` 函数改名为 `xy_to_rt` 和 `rt_to_xy`，并将 `radians` 参数修改为 `degrees`。
- 移除 `_shp` 模块里与二进制文件无关的函数。

## `0.3.6`

- 修改 `BinaryConverter` 的用法。
- 修改 `MANIFEST.in` 文件，纳入 `README.md` 文件。

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

- 新增 `_shp` 模块，改用自定义的二进制格式存储 shapefile 文件，减半文件体积的同时加快读取速度。
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