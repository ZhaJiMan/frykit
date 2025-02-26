from __future__ import annotations

from frykit.shp.data import (
    AdmKey,
    clear_data_cache,
    get_cn_border,
    get_cn_city,
    get_cn_city_names,
    get_cn_city_table,
    get_cn_district,
    get_cn_district_names,
    get_cn_district_table,
    get_cn_province,
    get_cn_province_names,
    get_cn_province_table,
    get_countries,
    get_land,
    get_nine_line,
    get_ocean,
)
from frykit.shp.mask import polygon_mask, polygon_mask2, polygon_to_mask
from frykit.shp.typing import PolygonType
from frykit.shp.utils import (
    EMPTY_PATH,
    EMPTY_POLYGON,
    GeometryTransformer,
    box_path,
    geom_to_path,
    geometry_to_dict,
    geometry_to_shape,
    get_geojson_geometries,
    get_geojson_properties,
    get_representative_xy,
    get_shapefile_geometries,
    get_shapefile_properties,
    make_feature,
    make_geojson,
    orient_polygon,
    path_to_polygon,
)
