import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import frykit.plot as fplt

ds = xr.load_dataset('data.nc')
t2m = ds['t2m'].isel(time=0) - 273.15
levels = np.linspace(10, 35, 26)

projs = [
    ccrs.PlateCarree(),
    ccrs.Mercator(),
    ccrs.LambertConformal(central_longitude=105, standard_parallels=(25, 47)),
    ccrs.Orthographic(central_longitude=105, central_latitude=30)
]
crs_data = ccrs.PlateCarree()

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(wspace=0.4, hspace=-0.2)
for i, crs_map in enumerate(projs):
    ax = fig.add_subplot(2, 2, i + 1, projection=crs_map)
    ax.coastlines(resolution='10m', lw=0.5)
    fplt.add_cn_border(ax, lw=0.5)
    fplt.set_extent_and_ticks(
        ax, extents=[70, 140, 0, 60],
        xticks=np.arange(50, 160 + 10, 20),
        yticks=np.arange(0, 60 + 10, 20),
        right=True, top=True,
        grid=True, lw=0.5, ls='--'
    )
    ax.tick_params(labelsize='x-small')
    cf = ax.contourf(
        t2m.longitude, t2m.latitude, t2m, levels,
        cmap='turbo', extend='both', transform=crs_data
    )
    fplt.clip_by_cn_border(cf)

cbar = fig.colorbar(
    cf, ax=fig.axes, pad=0.1,
    shrink=0.8, aspect=30, extendfrac=0.04
)
cbar.ax.tick_params(labelsize='x-small')
cbar.set_label('2m Temperature (℃)', fontsize='x-small')

fig.savefig('projections.png', dpi=300, bbox_inches='tight')
plt.close(fig)