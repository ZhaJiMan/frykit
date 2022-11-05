import json
from pathlib import Path

import shapefile

# 准备省界和市界的文件.
dirpath_shp = Path('../frykit/data/shp')
filepath_province = dirpath_shp / 'province.shp'
filepath_city = dirpath_shp / 'city.shp'

# 省名到省文件序号的映射.
pr_name_to_pr_index = {}
with shapefile.Reader(str(filepath_province)) as reader:
    for i, record in enumerate(reader.iterRecords()):
        pr_name = record['pr_name']
        pr_name_to_pr_index[pr_name] = i

# 市名和省名到市文件序号的映射.
ct_name_to_ct_index = {}
pr_name_to_ct_indices = {}
with shapefile.Reader(str(filepath_city)) as reader:
    for i, record in enumerate(reader.iterRecords()):
        ct_name = record['ct_name']
        pr_name = record['pr_name']
        ct_name_to_ct_index[ct_name] = i
        if pr_name in pr_name_to_ct_indices:
            pr_name_to_ct_indices[pr_name].append(i)
        else:
            pr_name_to_ct_indices[pr_name] = [i]

# 保存到json文件.
mapping = {
    'pr_name_to_pr_index': pr_name_to_pr_index,
    'ct_name_to_ct_index': ct_name_to_ct_index,
    'pr_name_to_ct_indices': pr_name_to_ct_indices
}

filepath_index = dirpath_shp / 'index.json'
with open(str(filepath_index), 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False)