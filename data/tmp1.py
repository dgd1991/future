import hashlib

import pandas as pd
path='E:/pythonProject/future/data/datafile/sample/model_v8/train_sample_2008.csv'
train_data_raw = pd.read_csv(path)
max = train_data_raw.max()
max = max.reset_index(level=0, drop=False)
max = max.reset_index(level=0, drop=False)
max.to_csv('E:/pythonProject/future/data/datafile/sample/model_v8/train_sample_2008_info_max.csv', mode='a',header=True, index=False, encoding='utf-8')
min = train_data_raw.min()
min = min.reset_index(level=0, drop=False)
min = min.reset_index(level=0, drop=False)
min.to_csv('E:/pythonProject/future/data/datafile/sample/model_v8/train_sample_2008_info_min.csv', mode='a',header=True, index=False, encoding='utf-8')

