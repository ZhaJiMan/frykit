"""
展示 frykit.plot 里 colorbar 相关函数的效果：

- make_qualitative_palette
- CenteredBoundaryNorm
- plot_colormap

需要安装 cmaps
"""

import cmaps
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import frykit.plot as fplt


def plot_qualitative_cbar(ax: Axes) -> None:
    """画出 make_qualitative_palette 的效果"""
    colors = [
        "orangered",
        "orange",
        "yellow",
        "limegreen",
        "royalblue",
        "darkviolet",
    ]
    cmap, norm, ticks = fplt.make_qualitative_palette(colors)
    cbar = fplt.plot_colormap(cmap, norm, ax=ax)
    cbar.set_ticks(ticks)  # pyright: ignore[reportArgumentType]
    cbar.set_ticklabels(colors)


def plot_centered_discrete_cbar(ax: Axes) -> None:
    """画出 CenteredBoundaryNorm 的效果"""
    boundaries = [-10, -5, -2, -1, 1, 2, 5, 10, 20, 50, 100]
    cmap = cmaps.BlueWhiteOrangeRed  # pyright: ignore[reportAttributeAccessIssue]
    norm = fplt.CenteredBoundaryNorm(boundaries)
    cbar = fplt.plot_colormap(cmap, norm, ax=ax)
    cbar.set_ticks(boundaries)


fig, axes = plt.subplots(2, 1, figsize=(8, 2))
fig.subplots_adjust(hspace=0.8)
plot_qualitative_cbar(axes[0])
plot_centered_discrete_cbar(axes[1])

fplt.savefig("colorbar.png")
plt.close(fig)
