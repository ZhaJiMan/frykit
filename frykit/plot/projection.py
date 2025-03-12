from cartopy.crs import AzimuthalEquidistant, Mercator, PlateCarree

__all__ = ["PLATE_CARREE", "WEB_MERCATOR", "CN_AZIMUTHAL_EQUIDISTANT"]

# 等经纬度投影
PLATE_CARREE = PlateCarree()

# 网络墨卡托投影
WEB_MERCATOR = Mercator.GOOGLE

# 竖版中国标准地图的投影
# http://gi.m.mnr.gov.cn/202103/t20210312_2617069.html
CN_AZIMUTHAL_EQUIDISTANT = AzimuthalEquidistant(
    central_longitude=105, central_latitude=35
)
