import math
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from cartopy.crs import Mercator, PlateCarree
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.axes._axes import Axes
from matplotlib.axes._base import _AxesBase
from matplotlib.backend_bases import RendererBase
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.quiver import Quiver, QuiverKey
from matplotlib.text import Text
from matplotlib.transforms import Affine2D, Bbox, ScaledTranslation, offset_copy

from frykit.calc import t_to_az


class QuiverLegend(QuiverKey):
    '''Quiver图例.'''

    def __init__(
        self,
        Q: Quiver,
        U: float,
        units: str = 'm/s',
        width: float = 0.15,
        height: float = 0.15,
        loc: Literal[
            'lower left', 'lower right', 'upper left', 'upper right'
        ] = 'lower right',
        qk_kwargs: Optional[dict] = None,
        patch_kwargs: Optional[dict] = None,
    ) -> None:
        self.units = units
        self.width = width
        self.height = height
        self.loc = loc

        if loc == 'lower left':
            X = width / 2
            Y = height / 2
        elif loc == 'lower right':
            X = 1 - width / 2
            Y = height / 2
        elif loc == 'upper left':
            X = width / 2
            Y = 1 - height / 2
        elif loc == 'upper right':
            X = 1 - width / 2
            Y = 1 - height / 2
        else:
            raise ValueError(
                "loc只能取{'lower left', 'lower right', 'upper left', 'upper right'}"
            )

        qk_kwargs = normalize_kwargs(qk_kwargs)
        super().__init__(
            Q=Q,
            X=X,
            Y=Y,
            U=U,
            label=f'{U} {units}',
            labelpos='S',
            coordinates='axes',
            **qk_kwargs,
        )
        # zorder必须在初始化后设置.
        zorder = qk_kwargs.get('zorder', 5)
        self.set_zorder(zorder)

        # 将qk调整至patch的中心.
        fontsize = self.text.get_fontsize() / 72
        dy = (self._labelsep_inches + fontsize) / 2
        trans = offset_copy(Q.axes.transAxes, Q.figure.figure, 0, dy)
        self.set_transform(trans)

        patch_kwargs = normalize_kwargs(patch_kwargs, Rectangle)
        patch_kwargs.setdefault('linewidth', 0.8)
        patch_kwargs.setdefault('edgecolor', 'k')
        patch_kwargs.setdefault('facecolor', 'w')
        self.patch = Rectangle(
            xy=(X - width / 2, Y - height / 2),
            width=width,
            height=height,
            transform=Q.axes.transAxes,
            **patch_kwargs,
        )

    def _set_transform(self) -> None:
        # 无效化QuiveKey的同名方法.
        pass

    def set_figure(self, fig: Figure) -> None:
        self.patch.set_figure(fig)
        super().set_figure(fig)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        self.patch.draw(renderer)
        super().draw(renderer)


# TODO: Geodetic配合Mercator.GOOGLE时速度很慢?
class Compass(PathCollection):
    '''地图指北针.'''

    def __init__(
        self,
        x: float,
        y: float,
        angle: Optional[float] = None,
        size: float = 20,
        style: Literal['arrow', 'star', 'circle'] = 'arrow',
        pc_kwargs: Optional[dict] = None,
        text_kwargs: Optional[dict] = None,
    ) -> None:
        self.x = x
        self.y = y
        self.angle = angle
        self.size = size
        self.style = style

        # 箭头方向朝上.
        head = size / 72
        if style == 'arrow':
            width = axis = head * 2 / 3
            verts1 = [(0, 0), (0, axis), (-width / 2, axis - head), (0, 0)]
            verts2 = [(0, 0), (width / 2, axis - head), (0, axis), (0, 0)]
            paths = [Path(verts1), Path(verts2)]
            colors = ['k', 'w']
        elif style == 'circle':
            width = axis = head * 2 / 3
            radius = head * 2 / 5
            theta = np.linspace(0, 2 * np.pi, 100)
            rx = radius * np.cos(theta)
            ry = radius * np.sin(theta) + head / 9
            verts1 = np.column_stack((rx, ry))
            verts2 = [(0, 0), (0, axis), (-width / 2, axis - head), (0, 0)]
            verts3 = [(0, 0), (width / 2, axis - head), (0, axis), (0, 0)]
            paths = [Path(verts1), Path(verts2), Path(verts3)]
            colors = ['none', 'k', 'w']
        elif style == 'star':
            width = head / 3
            axis = head + width / 2
            verts1 = [(0, 0), (0, axis), (-width / 2, axis - head), (0, 0)]
            verts2 = [(0, 0), (width / 2, axis - head), (0, axis), (0, 0)]
            path1 = Path(verts1)
            path2 = Path(verts2)
            paths = []
            for deg in range(0, 360, 90):
                rotation = Affine2D().rotate_deg(deg)
                paths.append(path1.transformed(rotation))
                paths.append(path2.transformed(rotation))
            colors = ['k', 'w']
        else:
            raise ValueError("style只能取{'arrow', 'circle', 'star'}")

        pc_kwargs = normalize_kwargs(pc_kwargs, PathCollection)
        pc_kwargs.setdefault('linewidth', 1)
        pc_kwargs.setdefault('edgecolor', 'k')
        pc_kwargs.setdefault('facecolor', colors)
        pc_kwargs.setdefault('clip_on', False)
        pc_kwargs.setdefault('zorder', 5)
        # 当pc_kwargs中也有transform时会报错.
        super().__init__(paths, transform=None, **pc_kwargs)

        # 文字在箭头上方.
        pad = head / 2.5
        text_kwargs = normalize_kwargs(text_kwargs, Text)
        text_kwargs.setdefault('fontsize', size / 1.5)
        self.text = Text(
            x=0,
            y=axis + pad,
            text='N',
            ha='center',
            va='center',
            rotation=0,
            transform=None,
            **text_kwargs,
        )

    def _init(self) -> None:
        # 计算指北针的方向.
        if self.angle is None:
            if isinstance(self.axes, GeoAxes):
                crs = PlateCarree()
                axes_to_data = self.axes.transAxes - self.axes.transData
                x0, y0 = axes_to_data.transform((self.x, self.y))
                crs.transform_point(x0, y0, self.axes.projection)
                lon0, lat0 = crs.transform_point(x0, y0, self.axes.projection)
                lon1, lat1 = lon0, min(lat0 + 0.01, 90)
                x1, y1 = self.axes.projection.transform_point(lon1, lat1, crs)
                theta = math.degrees(math.atan2(y1 - y0, x1 - x0))
                azimuth = t_to_az(theta, degrees=True)
            else:
                azimuth = 0
        else:
            azimuth = self.angle

        rotation = Affine2D().rotate_deg(-azimuth)
        translation = ScaledTranslation(self.x, self.y, self.axes.transAxes)
        trans = self.axes.figure.dpi_scale_trans + rotation + translation
        self.text.set_transform(trans)
        self.text.set_rotation(-azimuth)
        self.set_transform(trans)

    def set_figure(self, fig: Figure) -> None:
        self.text.set_figure(fig)
        super().set_figure(fig)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        self._init()
        self.text.draw(renderer)
        super().draw(renderer)


class ScaleBar(_AxesBase):
    '''地图比例尺.'''

    def __init__(
        self,
        ax: Axes,
        x: float,
        y: float,
        length: float = 1000,
        units: Literal['m', 'km'] = 'km',
    ) -> None:
        self.x = x
        self.y = y
        self.length = length
        self.units = units

        if units == 'm':
            self._units = 1
        elif units == 'km':
            self._units = 1000
        else:
            raise ValueError("units只能取{'m', 'km'}")

        # 避免全局rc设置影响刻度样式.
        with plt.style.context('default'):
            super().__init__(ax.figure, (0, 0, 1, 1), zorder=5)
        ax.add_child_axes(self)

        # 只显示上边框的刻度.
        self.set_xlabel(units, fontsize='medium')
        self.set_xlim(0, length)
        self.tick_params(
            which='both',
            left=False,
            labelleft=False,
            bottom=False,
            labelbottom=False,
            top=True,
            labeltop=True,
            labelsize='small',
        )

    def _init(self) -> None:
        # 在data坐标系取一条直线, 计算单位长度对应的地理长度.
        if isinstance(self.axes, GeoAxes):
            # GeoAxes取地图中心一段横线.
            crs = PlateCarree()
            geod = crs.get_geod()
            xmin, xmax = self.axes.get_xlim()
            ymin, ymax = self.axes.get_ylim()
            xmid = (xmin + xmax) / 2
            ymid = (ymin + ymax) / 2
            dx = (xmax - xmin) / 10
            x0 = xmid - dx / 2
            x1 = xmid + dx / 2
            lon0, lat0 = crs.transform_point(x0, ymid, self.axes.projection)
            lon1, lat1 = crs.transform_point(x1, ymid, self.axes.projection)
            dr = geod.inv(lon0, lat0, lon1, lat1)[2] / self._units
            dxdr = dx / dr
        else:
            # Axes认为是PlateCarree投影, 取中心纬度.
            Re = 6371e3
            L = 2 * np.pi * Re / 360
            lat0, lat1 = self.axes.get_ylim()
            lat = (lat0 + lat1) / 2
            drdx = L * math.cos(math.radians(lat))
            dxdr = self._units / drdx

        # 重新设置比例尺的大小和位置.
        axes_to_data = self.axes.transAxes - self.axes.transData
        data_to_figure = self.axes.transData - self.figure.transSubfigure
        x, y = axes_to_data.transform((self.x, self.y))
        width = self.length * dxdr
        bbox = Bbox.from_bounds(x, y, width, 1e-4 * width)
        bbox = data_to_figure.transform_bbox(bbox)
        self.set_position(bbox)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        self._init()
        super().draw(renderer)


def path_from_extents(
    x0: float, x1: float, y0: float, y1: float, ccw: bool = True
) -> Path:
    '''根据方框范围构造Path对象. ccw表示逆时针.'''
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    if not ccw:
        verts.reverse()
    path = Path(verts)

    return path


class Frame(Artist):
    '''GMT风格的边框.'''

    def __init__(self, width: float = 5, **kwargs: Any) -> None:
        self.width = width
        self._width = width / 72
        super().__init__()
        self.set_zorder(2.5)

        kwargs = normalize_kwargs(kwargs, PathCollection)
        kwargs.setdefault('linewidth', 1)
        kwargs.setdefault('edgecolor', 'k')
        kwargs.setdefault('facecolor', ['k', 'w'])

        # 暂时用空paths占位.
        self.pcs = {}
        for key in ['left', 'right', 'top', 'bottom', 'corner']:
            self.pcs[key] = PathCollection([], transform=None, **kwargs)

    def _init(self) -> None:
        if isinstance(self.axes, GeoAxes) and not isinstance(
            self.axes.projection, (PlateCarree, Mercator)
        ):
            raise ValueError('只支持PlateCarree和Mercator投影')

        # 将inches坐标系的width转为axes坐标系里的dx和dy.
        inches_to_axes = self.axes.figure.dpi_scale_trans - self.axes.transAxes
        mtx = inches_to_axes.get_matrix()
        dx = self._width * mtx[0, 0]
        dy = self._width * mtx[1, 1]

        xtrans = self.axes.get_xaxis_transform()
        ytrans = self.axes.get_yaxis_transform()
        self.pcs['top'].set_transform(xtrans)
        self.pcs['bottom'].set_transform(xtrans)
        self.pcs['left'].set_transform(ytrans)
        self.pcs['right'].set_transform(ytrans)
        self.pcs['corner'].set_transform(self.axes.transAxes)

        # 收集xlim和ylim范围内所有刻度并去重.
        major_xticks = self.axes.xaxis.get_majorticklocs()
        minor_xticks = self.axes.xaxis.get_minorticklocs()
        major_yticks = self.axes.yaxis.get_majorticklocs()
        minor_yticks = self.axes.yaxis.get_minorticklocs()
        xmin, xmax = sorted(self.axes.get_xlim())
        ymin, ymax = sorted(self.axes.get_ylim())
        xticks = np.array([xmin, xmax, *major_xticks, *minor_xticks])
        yticks = np.array([ymin, ymax, *major_yticks, *minor_yticks])
        xticks = xticks[(xticks >= xmin) & (xticks <= xmax)]
        yticks = yticks[(yticks >= ymin) & (yticks <= ymax)]
        # 通过round抵消central_longitude导致的浮点误差.
        xticks = np.unique(xticks.round(9))
        yticks = np.unique(yticks.round(9))
        nx = len(xticks)
        ny = len(yticks)

        top_paths = [
            path_from_extents(xticks[i], xticks[i + 1], 1, 1 + dy)
            for i in range(nx - 1)
        ]
        # 即便xaxis方向反转也维持边框的颜色顺序.
        if self.axes.xaxis.get_inverted():
            top_paths.reverse()
        self.pcs['top'].set_paths(top_paths)

        # 比例尺对象只设置上边框.
        if isinstance(self.axes, ScaleBar):
            return None

        bottom_paths = [
            path_from_extents(xticks[i], xticks[i + 1], -dy, 0)
            for i in range(nx - 1)
        ]
        if self.axes.xaxis.get_inverted():
            bottom_paths.reverse()
        self.pcs['bottom'].set_paths(bottom_paths)

        left_paths = [
            path_from_extents(-dx, 0, yticks[i], yticks[i + 1])
            for i in range(ny - 1)
        ]
        if self.axes.yaxis.get_inverted():
            left_paths.reverse()
        self.pcs['left'].set_paths(left_paths)

        right_paths = [
            path_from_extents(1, 1 + dx, yticks[i], yticks[i + 1])
            for i in range(ny - 1)
        ]
        if self.axes.yaxis.get_inverted():
            right_paths.reverse()
        self.pcs['right'].set_paths(right_paths)

        # 单独画出四个角落的方块.
        corner_paths = [
            path_from_extents(-dx, 0, -dy, 0),
            path_from_extents(1, 1 + dx, -dy, 0),
            path_from_extents(-dx, 0, 1, 1 + dy),
            path_from_extents(1, 1 + dx, 1, 1 + dy),
        ]
        self.pcs['corner'].set_paths(corner_paths)
        fc = self.pcs['top'].get_facecolor()[-1]
        self.pcs['corner'].set_facecolor(fc)

    def set_figure(self, fig: Figure) -> None:
        super().set_figure(fig)
        for pc in self.pcs.values():
            pc.set_figure(fig)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        self._init()
        for pc in self.pcs.values():
            pc.draw(renderer)
        self.stale = False
