import cmaps
import matplotlib.pyplot as plt

import frykit.plot as fplt


def plot_qualitative_cbar(ax):
    """画出 make_qualitative_cmap 的效果"""
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
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(colors)


def plot_centered_discrete_cbar(ax):
    """画出 CenteredBoundaryNorm 的效果"""
    boundaries = [-10, -5, -2, -1, 1, 2, 5, 10, 20, 50, 100]
    norm = fplt.CenteredBoundaryNorm(boundaries)
    cbar = fplt.plot_colormap(cmaps.BlueWhiteOrangeRed, norm, ax=ax)
    cbar.set_ticks(boundaries)


fig, axes = plt.subplots(2, 1, figsize=(8, 2))
fig.subplots_adjust(hspace=0.8)
plot_qualitative_cbar(axes[0])
plot_centered_discrete_cbar(axes[1])

fplt.savefig("../image/colorbar.png")
plt.close(fig)
