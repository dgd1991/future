import gc
import os
import numpy as np
from tools.Tools import Tools
import pandas as pd

data1 = pd.read_csv('E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_2008.csv')
data2 = pd.read_csv('E:/pythonProject/future/data/datafile/feature/model_v12/code_feature_2008.csv')
raw_k_data = data1[(data1['industry_id_level3'] > 0) | (data1['code'] == 'sh.000001') | (data1['code'] == 'sz.399001') | (data1['code'] == 'sz.399006')]
raw_k_data = raw_k_data[(raw_k_data['tradestatus'] == 1) & (raw_k_data['turn'] > 0) & (raw_k_data['pctChg'] < 21) & (raw_k_data['pctChg'] > -21)]
raw_k_data = raw_k_data.sort_values(['date'])
raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))
raw_k_data['is_new'] = raw_k_data["pctChg"].groupby(level=0).apply(lambda x: x.rolling(min_periods=20, window=20, center=False).apply(lambda y: y[0]))
raw_k_data = raw_k_data[raw_k_data['is_new'].map(lambda x: False if np.isnan(x) else True)]
raw_k_data = raw_k_data.reset_index(level=0, drop=True)
raw_k_data = raw_k_data.reset_index(level=0, drop=False)
raw_k_data = raw_k_data[raw_k_data['date']>=(str(2007)+"-01-01")]
data1 = raw_k_data

data1=data1[(data1['date']=='2008-01-03') & (data1['industry_id_level1']==1.0)][['code','industry_id_level1']]
data2=data2[(data2['date']=='2008-01-03') & (data2['industry_id_level1']==1.0)][['code','industry_id_level1']]
data3 = pd.merge(data1, data2, how="left", left_on=["code"],right_on=["code"])
print(data3)

# sh.600354
# sh.600311
# sh.600703
