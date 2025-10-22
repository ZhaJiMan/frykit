"""
- 通过 TypedDict 对 matplotlib 相关的 **kwargs 参数进行类型标注
- 因为 matplotlib 的参数太多，所以这里只列出常用字段，其它字段通过 extra_items 允许。
- 很多参数可以取 None，但一般 **kwargs 里不给出这个参数等同于取 None，所以这里省略了 None。
- 实际能接受的参数相比文档来说更加宽松，但参考 matplotlib 的 pyi 文件指定更严格的类型。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from cartopy.crs import Projection
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.font_manager import FontProperties
    from matplotlib.patheffects import AbstractPathEffect
    from matplotlib.transforms import Bbox, Transform
    from numpy.typing import ArrayLike

__all__ = [
    "AddBoxKwargs",
    "CompassPcKwargs",
    "CompassTextKwargs",
    "FrameKwargs",
    "GeometryPathCollectionKwargs",
    "LetterAxesKwargs",
    "QuiverLegendPatchKwargs",
    "QuiverLegendQkKwargs",
    "SaveFigKwargs",
    "TextCollectionKwargs",
]


# TODO: from matplotlib.typing import ColorType
RGBColorType: TypeAlias = tuple[float, float, float] | str
RGBAColorType: TypeAlias = (
    str
    | tuple[float, float, float, float]
    | tuple[RGBColorType, float]  # (color, alpha)
    | tuple[tuple[float, float, float, float], float]  # (4-tuple, alpha)
)
ColorType: TypeAlias = RGBColorType | RGBAColorType

LineStyleType: TypeAlias = str | tuple[float, Sequence[float]]

FontStyleType: TypeAlias = Literal["normal", "italic", "oblique"]
FontVariantType: TypeAlias = Literal["normal", "small-caps"]

HorizontalAlignmentType: TypeAlias = Literal["left", "center", "right"]
VerticalAlignmentType: TypeAlias = Literal[
    "baseline", "bottom", "center", "center_baseline", "top"
]
MultiAlignmentType: TypeAlias = Literal["left", "center", "right"]


class BaseArtistKwargs(TypedDict, total=False, extra_items=Any):
    alpha: float
    clip_on: bool
    zorder: float


class BasePathCollectionKwargs(BaseArtistKwargs, total=False):
    dashes: LineStyleType | Sequence[LineStyleType]
    ec: ColorType | Sequence[ColorType]
    edgecolor: ColorType | Sequence[ColorType]
    edgecolors: ColorType | Sequence[ColorType]
    fc: ColorType | Sequence[ColorType]
    facecolor: ColorType | Sequence[ColorType]
    facecolors: ColorType | Sequence[ColorType]
    linestyle: LineStyleType | Sequence[LineStyleType]
    linestyles: LineStyleType | Sequence[LineStyleType]
    linewidth: float | Sequence[float]
    linewidths: float | Sequence[float]
    ls: LineStyleType | Sequence[LineStyleType]
    lw: float | Sequence[float]
    path_effects: list[AbstractPathEffect]


class BasePatchKwargs(BaseArtistKwargs, total=False):
    ec: ColorType
    edgecolor: ColorType
    fc: ColorType
    facecolor: ColorType
    fill: bool
    linestyle: LineStyleType
    linewidth: float
    ls: LineStyleType
    lw: float
    path_effects: list[AbstractPathEffect]


class BaseTextKwargs(BaseArtistKwargs, total=False):
    backgroundcolor: ColorType
    bbox: dict[str, Any]
    c: ColorType
    color: ColorType
    family: str | Sequence[str]
    font: str | Path | FontProperties
    font_properties: str | Path | FontProperties
    fontfamily: str | Sequence[str]
    fontname: str | Sequence[str]
    fontproperties: str | Path | FontProperties
    fontsize: float | str
    fontstretch: int | str
    fontstyle: FontStyleType
    fontvariant: FontVariantType
    fontweight: int | str
    name: str | Sequence[str]
    path_effects: list[AbstractPathEffect]
    size: float | str
    stretch: int | str
    style: FontStyleType
    variant: FontVariantType
    weight: int | str


class GeometryPathCollectionKwargs(BasePathCollectionKwargs, total=False):
    array: ArrayLike
    cmap: str | Colormap
    norm: Normalize
    offset_transform: Transform
    offsets: tuple[float, float] | Sequence[tuple[float, float]]
    transform: Transform  # 不推荐 Projection


class TextCollectionKwargs(BaseTextKwargs, total=False):
    ha: HorizontalAlignmentType
    horizontalalignment: HorizontalAlignmentType
    linespacing: float
    ma: MultiAlignmentType
    math_fontfamily: str
    multialignment: MultiAlignmentType
    parse_math: bool
    rotation: float | Literal["vertical", "horizontal"]
    rotation_mode: Literal["default", "anchor"]
    transform: Transform | Projection | None  # 额外添加 Projection
    transform_rotates_text: bool
    va: VerticalAlignmentType
    verticalalignment: VerticalAlignmentType


class QuiverLegendQkKwargs(BaseArtistKwargs, total=False):
    color: ColorType
    fontproperties: dict[str, Any]
    labelcolor: ColorType
    labelsep: float


class QuiverLegendPatchKwargs(BasePatchKwargs, total=False):
    pass


class CompassPcKwargs(BasePathCollectionKwargs, total=False):
    pass


class CompassTextKwargs(BaseTextKwargs, total=False):
    pass


class FrameKwargs(BasePathCollectionKwargs, total=False):
    pass


class AddBoxKwargs(BasePatchKwargs, total=False):
    pass


class LetterAxesKwargs(BaseTextKwargs, total=False):
    pass


class SaveFigKwargs(TypedDict, total=False, extra_items=Any):
    bbox_inches: Literal["tight"] | Bbox
    dpi: float
    edgecolor: ColorType | Literal["auto"]
    facecolor: ColorType | Literal["auto"]
    format: str
    orientation: str
    pad_inches: float
    pil_kwargs: dict[str, Any]
    transparent: bool
