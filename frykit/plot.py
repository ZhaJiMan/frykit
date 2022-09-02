from pathlib import Path

import numpy as np
import shapely.geometry as sgeom
from pyproj import Geod
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.collections import PathCollection
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from PIL import Image

import frykit.shp as fshp

def set_extent_and_ticks(ax, **kwargs):
    '''设置GeoAxes的范围和刻度.'''
    if isinstance(ax.projection, (ccrs._RectangularProjection, ccrs.Mercator)):
        set_extent_and_ticks_rectangular(ax, **kwargs)
    else:
        set_extent_and_ticks_non_rectangular(ax, **kwargs)

def set_extent_and_ticks_rectangular(
    ax, extents=None,
    xticks=None, yticks=None, nx=0, ny=0,
    xformatter=None, yformatter=None,
    grid=False, **kwargs
):
    '''
    设置矩形投影的GeoAxes的范围和刻度.

    Parameters
    ----------
    ax : GeoAxes
        _RectangularProjection或Mercator投影的GeoAxes.

    extents : 4-tuple of float, optional
        经纬度范围[lonmin, lonmax, latmin, latmax]. 默认全球范围.

    xticks : array_like, optional
        经度主刻度的坐标. 默认不进行设置.

    yticks : array_like, optional
        纬度主刻度的坐标. 默认不进行设置.

    nx : int, optional
        经度主刻度之间次刻度的个数. 默认没有次刻度.
        当经度不是等距分布时, 请不要进行设置.

    ny : int, optional
        纬度主刻度之间次刻度的个数. 默认没有次刻度.
        当纬度不是等距分布时, 请不要进行设置.

    xformatter : Formatter, optional
        经度刻度标签的Formatter. 默认使用无参数的LongitudeFormatter.

    yformatter : Formatter, optional
        纬度刻度标签的Formatter. 默认使用无参数的LatitudeFormatter.

    grid : bool, optional
        是否沿主刻度绘制网格线.

    **kwargs
        绘制网格线的关键字参数.
        例如color, linewidth和linestyle等
    '''
    # 设置Formatter.
    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()

    # 设置主次刻度.
    crs = ccrs.PlateCarree()
    if xticks is not None:
        ax.set_xticks(xticks, crs=crs)
        if nx > 0:
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(nx + 1))
        ax.xaxis.set_major_formatter(xformatter)
    if yticks is not None:
        ax.set_yticks(yticks, crs=crs)
        if ny > 0:
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(ny + 1))
        ax.yaxis.set_major_formatter(yformatter)

    # 后调用set_extent, 防止刻度拓宽显示范围.
    if extents is None:
        ax.set_global()
    else:
        ax.set_extent(extents, crs=crs)

    # 绘制网格.
    if grid:
        ax.gridlines(crs, xlocs=xticks, ylocs=yticks, **kwargs)

def set_extent_and_ticks_non_rectangular(
    ax, extents,
    xticks=None, yticks=None,
    xformatter=None, yformatter=None,
    grid=False, **kwargs
):
    '''
    设置非矩形投影的GeoAxes的范围和刻度.

    Parameters
    ----------
    ax : GeoAxes
        非_RectangularProjection或Mercator投影的GeoAxes.

    extents : 4-tuple of float, optional
        经纬度范围[lonmin, lonmax, latmin, latmax].
        如果调用ax.set_extent后ax的范围不为矩形, 那么刻度位置会出错.

    xticks : array_like, optional
        经度主刻度的坐标. 默认不进行设置.

    yticks : array_like, optional
        纬度主刻度的坐标. 默认不进行设置.

    xformatter : Formatter, optional
        经度刻度标签的Formatter. 默认使用无参数的LongitudeFormatter.

    yformatter : Formatter, optional
        纬度刻度标签的Formatter. 默认使用无参数的LatitudeFormatter.

    grid : bool, optional
        是否沿主刻度绘制网格线.

    **kwargs
        绘制网格线的关键字参数.
        例如color, linewidth和linestyle等
    '''
    # 先设置范围, 使边框呈矩形.
    crs = ccrs.PlateCarree()
    ax.set_extent(extents, crs)

    # 获取更新后的范围.
    eps = 1
    npt = 100
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = np.linspace(x0, x1, npt)
    y = np.linspace(y0, y1, npt)
    X, Y = np.meshgrid(x, y)
    coords = crs.transform_points(ax.projection, X.ravel(), Y.ravel())
    lon, lat = coords[:, 0], coords[:, 1]
    lon0 = np.nanmin(lon) - eps
    lon1 = np.nanmax(lon) + eps
    lat0 = np.nanmin(lat) - eps
    lat1 = np.nanmax(lat) + eps

    # 设置Formatter.
    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()

    # 以经线与上下横轴的交点作为刻度.
    if xticks is not None:
        ticklocs_bottom, ticklocs_top = [], []
        ticklabels_bottom, ticklabels_top = [], []
        axis_bottom = sgeom.LineString([(x0, y0), (x1, y0)])
        axis_top = sgeom.LineString([(x0, y1), (x1, y1)])
        lat = np.linspace(lat0, lat1, npt)
        for xtick in xticks:
            lon = np.full_like(lat, xtick)
            line = sgeom.LineString(np.column_stack((lon, lat)))
            line = ax.projection.project_geometry(line, crs)
            point = axis_bottom.intersection(line)
            if isinstance(point, sgeom.Point) and not point.is_empty:
                ticklocs_bottom.append(point.x)
                ticklabels_bottom.append(xformatter(xtick))
            point = axis_top.intersection(line)
            if isinstance(point, sgeom.Point) and not point.is_empty:
                ticklocs_top.append(point.x)
                ticklabels_top.append(xformatter(xtick))

        # 让两个axis的刻度不同.
        ax.set_xticks(ticklocs_bottom + ticklocs_top)
        ax.set_xticklabels(ticklabels_bottom + ticklabels_top)
        ind = len(ticklabels_bottom)
        for tick in ax.xaxis.get_major_ticks()[:ind]:
            tick.tick2line.set_visible(False)
            tick.label2.set_visible(False)
        for tick in ax.xaxis.get_major_ticks()[ind:]:
            tick.tick1line.set_visible(False)
            tick.label1.set_visible(False)

    # 以纬线与左右纵轴的交点作为刻度.
    if yticks is not None:
        ticklocs_left, ticklocs_right = [], []
        ticklabels_left, ticklabels_right = [], []
        axis_left = sgeom.LineString([(x0, y0), (x0, y1)])
        axis_right = sgeom.LineString([(x1, y0), (x1, y1)])
        lon = np.linspace(lon0, lon1, npt)
        for ytick in yticks:
            lat = np.full_like(lon, ytick)
            line = sgeom.LineString(np.column_stack((lon, lat)))
            line = ax.projection.project_geometry(line, crs)
            point = axis_left.intersection(line)
            if isinstance(point, sgeom.Point) and not point.is_empty:
                ticklocs_left.append(point.y)
                ticklabels_left.append(yformatter(ytick))
            point = axis_right.intersection(line)
            if isinstance(point, sgeom.Point) and not point.is_empty:
                ticklocs_right.append(point.y)
                ticklabels_right.append(yformatter(ytick))

        # 让两个axis的刻度不同.
        ax.set_yticks(ticklocs_left + ticklocs_right)
        ax.set_yticklabels(ticklabels_left + ticklabels_right)
        ind = len(ticklabels_left)
        for tick in ax.yaxis.get_major_ticks()[:ind]:
            tick.tick2line.set_visible(False)
            tick.label2.set_visible(False)
        for tick in ax.yaxis.get_major_ticks()[ind:]:
            tick.tick1line.set_visible(False)
            tick.label1.set_visible(False)

    # 绘制网格.
    if grid:
        ax.gridlines(crs, xlocs=xticks, ylocs=yticks, **kwargs)

def add_box(ax, extents, **kwargs):
    '''
    在Axes上添加一个方框.

    Parameters
    ----------
    ax : Axes
        目标Axes.

    extents : 4-tuple of float
        方框范围[xmin, xmax, ymin, ymax].

    **kwargs
        创建Rectangle对象的关键字参数.
        例如linewidth, edgecolor, facecolor和transform等.
    '''
    xmin, xmax, ymin, ymax = extents
    dx = xmax - xmin
    dy = ymax - ymin
    patch = mpatches.Rectangle((xmin, ymin), dx, dy, **kwargs)
    ax.add_patch(patch)

def locate_sub_axes(ax_main, ax_sub, shrink=0.4, loc='bottom right'):
    '''
    将ax_sub等比例缩小, 并摆放在ax_main的角落位置.

    Parameters
    ----------
    ax_main : Axes
        用于参考的主Axes.

    ax_sub : Axes
        需要被调整的副Axes.
        若为GeoAxes, 需要调整前调用ax_sub.set_extent.

    shrink : float, optional
        缩小倍数. 默认为0.2.

    loc : {'bottom left', 'bottom right', 'top left', 'top right'}
        将ax_sub摆放在四个角落中的哪一个.
    '''
    bbox_main = ax_main.get_position()
    bbox_sub = ax_sub.get_position()
    # 使shrink=1时ax_main与ax_sub等宽或等高.
    if bbox_sub.width > bbox_sub.height:
        ratio = bbox_main.width / bbox_sub.width * shrink
    else:
        ratio = bbox_main.height / bbox_sub.height * shrink
    width = bbox_sub.width * ratio
    height = bbox_sub.height * ratio

    # 可选四个角落位置.
    if loc == 'bottom left':
        x0 = bbox_main.x0
        x1 = bbox_main.x0 + width
        y0 = bbox_main.y0
        y1 = bbox_main.y0 + height
    elif loc == 'bottom right':
        x0 = bbox_main.x1 - width
        x1 = bbox_main.x1
        y0 = bbox_main.y0
        y1 = bbox_main.y0 + height
    elif loc == 'top left':
        x0 = bbox_main.x0
        x1 = bbox_main.x0 + width
        y0 = bbox_main.y1 - height
        y1 = bbox_main.y1
    elif loc == 'top right':
        x0 = bbox_main.x1 - width
        x1 = bbox_main.x1
        y0 = bbox_main.y1 - height
        y1 = bbox_main.y1
    else:
        raise ValueError('loc参数错误')
    bbox_new = mtransforms.Bbox.from_extents(x0, y0, x1, y1)
    ax_sub.set_position(bbox_new)

def add_side_axes(ax_main, loc, pad, depth):
    '''
    在原有的Axes旁边新添一个等高或等宽的Axes并返回该对象.

    Parameters
    ----------
    ax_main : Axes or array_like of Axes
        原有的Axes, 也可以是一组Axes构成的数组.

    loc : {'left', 'right', 'bottom', 'top'}
        新Axes相对于旧Axes的位置.

    pad : float
        新旧Axes的间距.

    depth : float
        新Axes的宽度或高度.

    Returns
    -------
    ax_side : Axes
        新Axes对象.
    '''
    # 获取一组Axes的位置.
    axes = np.atleast_1d(ax_main).ravel()
    bbox_main = mtransforms.Bbox.union([ax.get_position() for ax in axes])

    # 可选四个方向.
    if loc == 'left':
        x0 = bbox_main.x0 - pad - depth
        x1 = x0 + depth
        y0 = bbox_main.y0
        y1 = bbox_main.y1
    elif loc == 'right':
        x0 = bbox_main.x1 + pad
        x1 = x0 + depth
        y0 = bbox_main.y0
        y1 = bbox_main.y1
    elif loc == 'bottom':
        x0 = bbox_main.x0
        x1 = bbox_main.x1
        y0 = bbox_main.y0 - pad - depth
        y1 = y0 + depth
    elif loc == 'top':
        x0 = bbox_main.x0
        x1 = bbox_main.x1
        y0 = bbox_main.y1 + pad
        y1 = y0 + depth
    else:
        raise ValueError('loc参数错误')
    bbox_side = mtransforms.Bbox.from_extents(x0, y0, x1, y1)
    ax_side = axes[0].figure.add_axes(bbox_side)

    return ax_side

def get_slice_xticks(
    lon, lat, ntick=6, decimals=2,
    lon_formatter=None, lat_formatter=None
):
    '''
    返回垂直剖面图所需的横坐标, 刻度位置和刻度标签.

    用经纬度数组的点数表示横坐标, 在横坐标上取ntick个等距的刻度,
    利用线性插值计算每个刻度标签的经纬度值.

    Parameters
    ----------
    lon : (npt,) array_like
        剖面对应的经度数组.

    lat : (npt,) array_like
        剖面对应的纬度数组.

    ntick : int, optional
        刻度的数量. 默认取6个.

    decimals : int, optional
        刻度标签的小数位数. 默认保留两位小数.

    lon_formatter : Formatter, optional
        刻度标签里经度的Formatter. 默认使用无参数的LongitudeFormatter.

    lat_formatter : Formatter, optional
        刻度标签里纬度的Formatter. 默认使用无参数的LatitudeFormatter.

    Returns
    -------
    x : (npt,) ndarray
        剖面数据的横坐标. 数值等于np.arange(npt).

    xticks : (ntick,) ndarray
        横坐标的刻度位置.

    xticklabels : (ntick,) list of str
        横坐标的刻度标签. 用刻度处的经纬度值表示.
    '''
    # 线性插值计算刻度的经纬度值.
    npt = len(lon)
    x = np.arange(npt)
    xticks = np.linspace(0, npt - 1, ntick)
    lon_ticks = np.interp(xticks, x, lon).round(decimals)
    lat_ticks = np.interp(xticks, x, lat).round(decimals)

    # 获取字符串形式的刻度标签.
    xticklabels = []
    if lon_formatter is None:
        lon_formatter = LongitudeFormatter()
    if lat_formatter is None:
        lat_formatter = LatitudeFormatter()
    for i in range(ntick):
        lon_label = lon_formatter(lon_ticks[i])
        lat_label = lat_formatter(lat_ticks[i])
        xticklabels.append(lon_label + '\n' + lat_label)

    return x, xticks, xticklabels

def make_qualitative_cmap(colors):
    '''
    创建一组定性的colormap和norm, 同时返回刻度位置.

    Parameters
    ----------
    colors : (N,) list or (N, 3) or (N, 4) array_like
        colormap所含的颜色. 可以为含有颜色的列表或RGB(A)数组.

    Returns
    -------
    cmap : ListedColormap
        创建的colormap.

    norm : Normalize
        创建的norm. N个颜色对应于0~N-1范围的数据.

    ticks : (N,) ndarray
        colorbar刻度的坐标.
    '''
    N = len(colors)
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=-0.5, vmax=N-0.5)
    ticks = np.arange(N)

    return cmap, norm, ticks

def get_aod_cmap():
    '''返回适用于AOD的cmap.'''
    filepath = Path(__file__).parent / 'data' / 'NEO_modis_aer_od.csv'
    rgb = np.loadtxt(str(filepath), delimiter=',') / 256
    cmap = mcolors.ListedColormap(rgb)

    return cmap

def letter_axes(axes, x, y, **kwargs):
    '''
    给一组Axes按顺序标注字母.

    Parameters
    ----------
    axes : array_like of Axes
        目标Axes的数组.

    x : float or array_like
        字母的横坐标, 基于Axes单位.
        可以为标量或数组, 数组形状需与axes相同.

    y : float or array_like
        字母的纵坐标. 基于Axes单位.
        可以为标量或数组, 数组形状需与axes相同.

    y : float or array_like
        可以为标量或数组, 数组形状需与axes相同.

    **kwargs
        调用text时的关键字参数.
        例如fontsize, fontfamily和color等.
    '''
    axes = np.atleast_1d(axes)
    x = np.full_like(axes, x) if np.isscalar(x) else np.asarray(x)
    y = np.full_like(axes, y) if np.isscalar(y) else np.asarray(y)
    for i, (ax, xi, yi) in enumerate(zip(axes.flat, x.flat, y.flat)):
        letter = chr(97 + i)
        ax.text(
            xi, yi, f'({letter})', ha='center', va='center',
            transform=ax.transAxes, **kwargs
        )

def add_polygons(ax, polygons, crs=None, **kwargs):
    '''
    将多边形添加到Axes上.

    与GeoAxes.add_geometries的区别是坐标变换更快, 能返回PathCollection对象.
    若画图结果有误, 可以考虑传入transform参数.

    Parameters
    ----------
    ax : Axes or GeoAxes
        目标Axes.

    polygons : list of Polygon or MultiPolygon
        多边形构成的列表.

    crs : CRS, optional
        多边形所处的坐标系. 当crs不为None时, 认为ax是GeoAxes,
        会将polygons从crs变换到ax.projection坐标系上.

    **kwargs
        创建PathCollection时的关键字参数.
        例如facecolor, edgecolor, cmap, norm和array等.

    Returns
    -------
    pc : PathCollection
        代表Path的集合对象.
    '''
    if crs is not None:
        polygons = fshp.transform_geometries(polygons, crs, ax.projection)

    array = kwargs.get('array', None)
    if array is not None and len(array) != len(polygons):
        raise ValueError('array的长度与polygons不匹配')

    paths = [fshp.polygon_to_path(polygon) for polygon in polygons]
    pc = PathCollection(paths, **kwargs)
    ax.add_collection(pc)

    return pc

def _set_path_kwargs(kwargs):
    '''初始化绘制PathCollection的参数'''
    if not any(kw in kwargs for kw in ['facecolor', 'facecolors', 'fc']):
        kwargs['facecolor'] = 'none'
    if not any(kw in kwargs for kw in ['edgecolor', 'edgecolors', 'ec']):
        kwargs['edgecolor'] = 'black'
    kwargs.setdefault('zorder', 1.5)

def _select_shp_crs(ax):
    '''根据ax的类型为fshp.get_cnshp的几何对象选择坐标系.'''
    return ccrs.PlateCarree() if isinstance(ax, GeoAxes) else None

def add_cn_border(ax, **kwargs):
    '''向Axes或GeoAxes添加中国国界.'''
    _set_path_kwargs(kwargs)
    country = fshp.get_cnshp(level='国')
    add_polygons(ax, [country], _select_shp_crs(ax), **kwargs)

def add_cn_province(ax, name=None, **kwargs):
    '''向Axes或GeoAxes添加中国省界. 默认画出所有省.'''
    _set_path_kwargs(kwargs)
    provinces = fshp.get_cnshp(level='省', province=name)
    if not isinstance(provinces, list):
        provinces = [provinces]
    add_polygons(ax, provinces, _select_shp_crs(ax), **kwargs)

def add_nine_line(ax, **kwargs):
    '''向Axes或GeoAxes添加九段线.'''
    _set_path_kwargs(kwargs)
    nine_line = fshp.get_nine_line()
    add_polygons(ax, [nine_line], _select_shp_crs(ax), **kwargs)

def clip_by_polygon(artist, polygon, crs=None, fix=False):
    '''
    利用多边形裁剪Artist, 只显示多边形内的内容.

    Parameters
    ----------
    artist : Artist
        被裁剪的Artist对象. 可以返回自以下方法:
        - contour, contourf
        - pcolor, pcolormesh
        - imshow
        - quiver
        - scatter

    polygon : Polygon or MultiPolygon
        用于裁剪的多边形.

    crs : CRS, optional
        多边形所处的坐标系. 当crs不为None时, 认为ax是GeoAxes,
        会将polygons从crs变换到ax.projection坐标系上.

    fix : bool, optional
        是否修正裁剪后Artist会超出GeoAxes的问题. 默认不修正.
    '''
    ax = artist.axes
    if crs is not None:
        polygon = fshp.transform_geometry(polygon, crs, ax.projection)
    if fix and isinstance(ax, GeoAxes):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        polygon = polygon & sgeom.box(x0, y0, x1, y1)
    path = fshp.polygon_to_path(polygon)

    if hasattr(artist, 'collections'):
        for collection in artist.collections:
            collection.set_clip_path(path, ax.transData)
    else:
        artist.set_clip_path(path, ax.transData)

def clip_by_cn_border(artist, fix=False):
    '''用中国国界裁剪Artist.'''
    country = fshp.get_cnshp(level='国')
    clip_by_polygon(artist, country, _select_shp_crs(artist.axes), fix)

def _create_kwargs(kwargs):
    '''创建参数字典.'''
    return {} if kwargs is None else kwargs.copy()

def add_quiver_legend(
    Q, U, units='m/s',
    width=0.15, height=0.15, loc='bottom right',
    patch_kwargs=None, key_kwargs=None
):
    '''
    为Axes.quiver的结果添加图例.

    图例由背景方框patch和风箭头key组成.
    key下方有形如'{U} {units}'的标签.

    Parameters
    ----------
    Q : Quiver
        Axes.quiver返回的结果.

    U : float
        key的长度.

    units : str, optional
        key标签的单位. 默认为'm/s'.

    width : float, optional
        方框的宽度. 基于Axes坐标, 默认为0.15

    height : float, optional
        方框的高度. 基于Axes坐标, 默认为0.15

    loc : {'bottom left', 'bottom right', 'top left', 'top right'}
        将图例摆放在四个角落中的哪一个.

    patch_kwargs : dict, optional
        方框的参数. 例如facecolor, edgecolor, linewidth等.

    key_kwargs : dict, optional
        传给quiverkey的参数. 例如labelsep, fontproperties等.
    '''
    # 决定legend的位置.
    if loc == 'bottom left':
        x = width / 2
        y = height / 2
    elif loc == 'bottom right':
        x = 1 - width / 2
        y = height / 2
    elif loc == 'top left':
        x = width / 2
        y = 1 - height / 2
    elif loc == 'top right':
        x = 1 - width / 2
        y = 1 - height / 2
    else:
        raise ValueError('loc参数错误')

    # 初始化参数.
    patch_kwargs = _create_kwargs(patch_kwargs)
    if 'facecolor' not in patch_kwargs and 'fc' not in patch_kwargs:
        patch_kwargs['facecolor'] = 'white'
    if 'edgecolor' not in patch_kwargs and 'ec' not in patch_kwargs:
        patch_kwargs['edgecolor'] = 'black'
    if 'linewidth' not in patch_kwargs and 'lw' not in patch_kwargs:
        patch_kwargs['linewidth'] = 0.8
    patch_kwargs.setdefault('zorder', 3)
    key_kwargs = _create_kwargs(key_kwargs)

    # 在ax上添加patch.
    ax = Q.axes
    patch = mpatches.Rectangle(
        (x - width / 2, y - height / 2), width, height,
        transform=ax.transAxes, **patch_kwargs
    )
    ax.add_patch(patch)

    # 先创建QuiverKey对象.
    qk = ax.quiverkey(
        Q, X=x, Y=y, U=U, label=f'{U} {units}',
        labelpos='S', **key_kwargs
    )
    # 在参数中设置zorder无效.
    zorder = key_kwargs.get('zorder', 3)
    qk.set_zorder(zorder)

    # 再将qk调整至patch的中心.
    fontsize = qk.text.get_fontsize() / 72
    dy = (qk._labelsep_inches + fontsize) / 2
    transform = mtransforms.offset_copy(ax.transAxes, ax.figure, 0, dy)
    qk._set_transform = lambda: None  # 无效类方法.
    qk.set_transform(transform)

def add_north_arrow(ax, xy, length=20, path_kwargs=None, text_kwargs=None):
    '''
    向Axes添加指北针

    Parameters
    ----------
    ax : Axes
        目标Axes.

    xy : 2-tuple of float
        指北针的坐标. 基于Axes坐标系.

    length : float, optional
        指北针箭头的长度. 单位为点.

    path_kwargs : dict, optional
        指北针箭头的关键字参数.
        例如facecolors, edgecolors, linewidth等.

    text_kwargs : dict, optional
        绘制指北针N字的关键字参数.
        例如fontsize, fontweight和fontfamily等.
    '''
    # 初始化箭头参数.
    path_kwargs = _create_kwargs(path_kwargs)
    if not any(kw in path_kwargs for kw in ['facecolor', 'facecolors', 'fc']):
        path_kwargs['facecolors'] = ['k', 'w']
    if not any(kw in path_kwargs for kw in ['edgecolor', 'edgecolors', 'ec']):
        path_kwargs['edgecolors'] = 'k'
    if not any(kw in path_kwargs for kw in ['linewidth', 'linewidths', 'lw']):
        path_kwargs['linewidths'] = 1
    path_kwargs.setdefault('zorder', 3)
    path_kwargs.setdefault('clip_on', False)

    # 初始化文字参数.
    text_kwargs = _create_kwargs(text_kwargs)
    if 'fontsize' not in text_kwargs and 'size' not in text_kwargs:
        text_kwargs['fontsize'] = length / 1.5

    # 绘制箭头.
    x, y = xy
    offset = mtransforms.ScaledTranslation(x, y, ax.transAxes)
    transform = ax.figure.dpi_scale_trans + offset
    len_inches = length / 72
    width = axis = len_inches * 2 / 3
    verts_left = [(0, 0), (-width / 2, -len_inches), (0, -axis), (0, 0)]
    verts_right = [(0, 0), (0, -axis), (width / 2, -len_inches), (0, 0)]
    paths = [mpath.Path(verts_left), mpath.Path(verts_right)]
    collection = PathCollection(paths, transform=transform, **path_kwargs)
    ax.add_collection(collection)

    # 添加文字.
    pad = len_inches / 10
    ax.text(
        0, pad, 'N', ha='center', va='bottom',
        transform=transform, **text_kwargs
    )

def add_map_scale(
    ax, xy, length=1000, ticks=None, ticklength=5,
    line_kwargs=None, text_kwargs=None
):
    '''
    向GeoAxes添加地图比例尺.

    Parameters
    ----------
    ax : GeoAxes
        目标GeoAxes.

    xy : 2-tuple of float
        比例尺的中心坐标. 基于Axes坐标系.

    length : float, optional
        比例尺的总长度. 默认为1000km.

    ticks : list of float, optional
        刻度位置. 单位为km, 默认取[0, length]作为刻度.
        超出这一范围的刻度不会被画出.

    ticklength : float, optional
        刻度长度. 单位为点, 默认为5.

    line_kwargs
        绘制比例尺线条的关键字参数.

    text_kwargs
        绘制刻度标签文字的关键字参数.
    '''
    # 初始化线条参数.
    line_kwargs = _create_kwargs(line_kwargs)
    if 'linewidth' not in line_kwargs and 'lw' not in line_kwargs:
        line_kwargs['linewidth'] = 1.2
    if 'color' not in line_kwargs and 'c' not in line_kwargs:
        line_kwargs['color'] = 'k'
    line_kwargs.setdefault('zorder', 3)
    line_kwargs.setdefault('clip_on', False)

    # 初始化文字参数.
    text_kwargs = _create_kwargs(text_kwargs)
    if 'fontsize' not in text_kwargs and 'size' not in text_kwargs:
        text_kwargs['fontsize'] = 8

    # 取地图中心的水平线计算单位投影坐标的长度.
    crs = ccrs.PlateCarree()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    dx = (xmax - xmin)
    x0 = xmid - dx / 2
    x1 = xmid + dx / 2
    lon0, lat0 = crs.transform_point(x0, ymid, ax.projection)
    lon1, lat1 = crs.transform_point(x1, ymid, ax.projection)
    g = Geod(ellps='WGS84')
    dr = g.inv(lon0, lat0, lon1, lat1)[2] / 1000
    dxdr = dx / dr

    # xy转为data坐标.
    axes_to_data = ax.transAxes - ax.transData
    x, y = axes_to_data.transform(xy)
    dx = length * dxdr
    x0 = x - dx / 2
    x1 = x + dx / 2

    # 计算刻度位置.
    if ticks is None:
        ticks = [0, length]
    else:
        ticks = [tick for tick in ticks if 0 <= tick <= length]
    locs = np.array(ticks) * dxdr + x - dx / 2

    # 绘制比例尺.
    ticklength = ticklength / 72
    labelpad = ticklength / 2
    yrev = -line_kwargs['linewidth'] / 72 / 2
    offset = mtransforms.ScaledTranslation(0, y, ax.transData)
    transform = mtransforms.blended_transform_factory(
        ax.transData, ax.figure.dpi_scale_trans + offset
    )
    ax.hlines(y, x0, x1, transform=ax.transData, **line_kwargs)
    ax.vlines(locs, yrev, ticklength, transform=transform, **line_kwargs)
    for tick, loc in zip(ticks, locs):
        ax.text(
            loc, ticklength + labelpad, tick, ha='center', va='bottom',
            transform=transform, **text_kwargs
        )
    # 添加单位标签.
    ax.text(
        x, -1.5 * labelpad, 'km', ha='center', va='top',
        transform=transform, **text_kwargs
    )

def make_gif(
    filepaths_img, filepath_gif,
    duration=500, loop=0, optimize=False
):
    '''
    制作GIF图.

    Parameters
    ----------
    filepaths_img : list of str or Path
        图片路径的列表. 要求至少含两个元素.

    filepath_gif : str or Path
        输出GIF图片的路径.

    duration : int or list or tuple, optional
        每一帧的持续时间, 以毫秒为单位. 也可以用列表或元组分别指定每一帧的持续时间.
        默认为500ms=0.5s.

    loop : int, optional
        GIF图片循环播放的次数. 默认无限循环.

    optimize : bool, optional
        尝试压缩GIF图片的调色板.
    '''
    if len(filepaths_img) < 2:
        raise ValueError('至少需要两张图片')

    images = [Image.open(str(filepath)) for filepath in filepaths_img]
    images[0].save(
        str(filepath_gif), save_all=True, append_images=images[1:],
        duration=duration, loop=loop, optimize=optimize
    )