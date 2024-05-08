import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.path import Path

import frykit.plot as fplt

extents = [100, 125, 15, 40]
lon0, lon1, lat0, lat1 = extents

path = Path(
    [
        (lon0, lat0),
        (lon1, lat0),
        (lon1, lat1),
        (lon0, lat1),
        (lon0, lat0),
    ]
).interpolated(100)

data = fplt.load_test_data()
lon = data['longitude']
lat = data['latitude']
t2m = data['t2m']

crs = ccrs.PlateCarree()
ax = plt.axes(projection=fplt.CN_AZIMUTHAL_EQUIDISTANT)
fplt.add_cn_province(ax)
ax.set_extent(extents, crs=crs)
ax.set_boundary(path, transform=crs)
ax.gridlines(draw_labels=True, rotate_labels=False, color='k', ls='--')

# strict 参数适用于不规则的边界
pc = ax.pcolormesh(lon, lat, t2m, transform=crs)
fplt.clip_by_cn_border(pc, strict=True)

plt.savefig('../image/strict_clip.png', dpi=300, bbox_inches='tight')
plt.close()
