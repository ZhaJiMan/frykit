from typing import Literal, Optional

from matplotlib.artist import allow_rasterization
from matplotlib.backend_bases import RendererBase
from matplotlib.cbook import normalize_kwargs
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.quiver import Quiver, QuiverKey
from matplotlib.transforms import offset_copy


class QuiverLegend(QuiverKey):
    """
    Quiver图例.

    图例由背景方框patch和风箭头key组成,
    key下方有形如'{U} {units}'的标签.
    """

    def __init__(
        self,
        Q: Quiver,
        U: float,
        units: str = "m/s",
        width: float = 0.15,
        height: float = 0.15,
        loc: Literal[
            "bottom left", "bottom right", "top left", "top right"
        ] = "bottom right",
        quiver_key_kwargs: Optional[dict] = None,
        patch_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        Q : Quiver
            Axes.quiver返回的对象.

        U : float
            箭头长度.

        units : str, optional
            标签单位. 默认为m/s.

        width : float, optional
            图例宽度. 基于Axes坐标, 默认为0.15

        height : float, optional
            图例高度. 基于Axes坐标, 默认为0.15

        loc : {'bottom left', 'bottom right', 'top left', 'top right'}, optional
            图例位置. 默认为'bottom right'.

        quiver_key_kwargs : dict, optional
            QuiverKey类的关键字参数.
            例如labelsep, labelcolor, fontproperties等.
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiverkey.html

        patch_kwargs : dict, optional
            表示背景方框的Retangle类的关键字参数.
            例如linewidth, edgecolor, facecolor等.
            https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
        """
        if loc == "bottom left":
            X = width / 2
            Y = height / 2
        elif loc == "bottom right":
            X = 1 - width / 2
            Y = height / 2
        elif loc == "top left":
            X = width / 2
            Y = 1 - height / 2
        elif loc == "top right":
            X = 1 - width / 2
            Y = 1 - height / 2
        else:
            raise ValueError("loc参数错误")

        quiver_key_kwargs = normalize_kwargs(quiver_key_kwargs)
        patch_kwargs = normalize_kwargs(patch_kwargs, Rectangle)
        patch_kwargs.setdefault("linewidth", 0.8)
        patch_kwargs.setdefault("edgecolor", "k")
        patch_kwargs.setdefault("facecolor", "w")

        super().__init__(
            Q=Q,
            X=X,
            Y=Y,
            U=U,
            label=f"{U} {units}",
            labelpos="S",
            coordinates="axes",
            **quiver_key_kwargs,
        )
        self.set_zorder(5)

        # 将QuiverKey调整至patch的中心.
        fontsize = self.text.get_fontsize() / 72
        dy = (self._labelsep_inches + fontsize) / 2
        trans = offset_copy(Q.axes.transAxes, Q.figure.figure, 0, dy)
        self._set_transform = lambda: None
        self.set_transform(trans)

        self.patch = Rectangle(
            xy=(X - width / 2, Y - height / 2),
            width=width,
            height=height,
            transform=Q.axes.transAxes,
            **patch_kwargs,
        )

    def set_figure(self, fig: Figure) -> None:
        self.patch.set_figure(fig)
        super().set_figure(fig)

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        self.patch.draw(renderer)
        super().draw(renderer)


class Compass:
    pass


class ScaleBar:
    pass


class Frame:
    pass
