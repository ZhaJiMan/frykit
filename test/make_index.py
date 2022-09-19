import json
from pathlib import Path

import shapefile

# 准备省界和市界的文件.
dirpath_shp = Path('../frykit/data/shp')
filepath_province = dirpath_shp / 'province.shp'
filepath_city = dirpath_shp / 'city.shp'

# 省界的索引.
dict_province = {}
with shapefile.Reader(str(filepath_province)) as reader:
    for i, record in enumerate(reader.iterRecords()):
        dict_province[record['pr_name']] = i

# 市界的索引.
dict_by_pr = {}
dict_by_ct = {}
with shapefile.Reader(str(filepath_city)) as reader:
    for i, record in enumerate(reader.iterRecords()):
        dict_by_ct[record['ct_name']] = i
        if record['pr_name'] in dict_by_pr:
            dict_by_pr[record['pr_name']].append(i)
        else:
            dict_by_pr[record['pr_name']] = [i]

# 保存到json文件.
dict_city = {'by_pr': dict_by_pr, 'by_ct': dict_by_ct}
dict_index = {'province': dict_province, 'city': dict_city}
filepath_index = dirpath_shp / 'index.json'
with open(str(filepath_index), 'w', encoding='utf-8') as f:
    json.dump(dict_index, f, ensure_ascii=False)