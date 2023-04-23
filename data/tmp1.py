import hashlib

import pandas as pd
from tools.path_enum import Path
pd.set_option('display.max_columns', 200)
# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)

sw_code_all_industry_path = Path.sw_code_all_industry_path
path='E:/pythonProject/future/data/datafile/industry/industry_id_level3_2021.csv'
train_data_raw = pd.read_csv(path)
code_industry = pd.read_csv(sw_code_all_industry_path, encoding='utf-8')
code_industry['start_date'] = pd.to_datetime(code_industry['start_date'])
code_industry['row_num'] = code_industry.groupby(['code'])['start_date'].rank(ascending=False, method='first').astype(int)
code_industry = code_industry[code_industry['row_num'] == 1]
code_industry = code_industry[['industry_id_level3', 'industry_name_level2']]
code_industry.drop_duplicates(inplace=True)
train_data_raw = pd.merge(train_data_raw, code_industry, how="inner", left_on=['industry_id_level3'], right_on=['industry_id_level3'])

train_data_raw[['industry_name_level2','date','industry_id_level3_close']]
train_data_raw= train_data_raw.groupby('industry_name_level2').apply(lambda x: x.set_index('date'))
train_data_raw['pctchg'] = train_data_raw['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1]-y[0])/y[0]))
print(train_data_raw['pctchg'].max())
print(train_data_raw['pctchg'].min())
# tmp=train_data_raw[train_data_raw['pctchg']>0.12]
# print(tmp)
