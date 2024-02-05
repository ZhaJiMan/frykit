import numpy as np
import matplotlib.pyplot as plt
import frykit.plot as fplt
import cmaps

def plot_qualitative_cbar(ax):
    '''画出make_qualitative_cmap的效果.'''
    colors = [
        'orangered', 'orange', 'yellow',
        'limegreen', 'royalblue', 'darkviolet'
    ]
    cmap, norm, ticks = fplt.get_qualitative_palette(colors)
    cbar = fplt.plot_colormap(cmap, norm, ax=ax)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(colors)

def plot_centered_discrete_cbar(ax):
    '''画出CenteredBoundaryNorm的效果.'''
    boundaries = [-10, -5, -2, -1, 1, 2, 5, 10, 20, 50, 100]
    norm = fplt.CenteredBoundaryNorm(boundaries)
    cbar = fplt.plot_colormap(cmaps.BlueWhiteOrangeRed, norm, ax=ax)
    cbar.set_ticks(boundaries)

fig, axes = plt.subplots(2, 1, figsize=(8, 2))
fig.subplots_adjust(hspace=0.8)
plot_qualitative_cbar(axes[0])
plot_centered_discrete_cbar(axes[1])

fig.savefig('../image/colorbar.png', dpi=300, bbox_inches='tight')
plt.close(fig)