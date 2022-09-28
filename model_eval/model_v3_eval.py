import numpy as np
import pandas as pd

year = 2020
model_name = 'model_v3'
df = pd.read_csv('E:/pythonProject/stock/data/datafile/prediction_result/{model_name}/prediction_result_{year}.csv'.format(model_name=model_name, year=str(year)), encoding='utf-8')
df.columns = ['date', 'code', 'predition_7d', 'label_1d', 'label_3d', 'label_5d', 'label_7d']
# df = df.head(1000)
df['date'] = df['date'].map(lambda x: x.replace('b', ''))
df['code'] = df['code'].map(lambda x: x.replace('b', '').replace("'", ''))
df['date'] = pd.to_datetime(df['date'])

# df = df[df['date'] == '2015-02-05']
# df = df.sort_values(['prediction'], ascending=False)
# print(df.head(10))
# print(df.head(10)['label'].mean())

all_df = df[df['code'] == "sh.000001"]
df_7d = df.sort_values(['date', 'predition_7d'], ascending=[True, False]).groupby('date').head(20)
# print(df_7d.isnull().any(axis=1))
# # print(df_7d[type(df_7d['label_7d']) == type(np.nan())])
# print(df_7d.info())
all_mean = all_df.mean()
all_code_mean = df[['label_1d','label_3d','label_5d','label_7d']].mean()

# label_1d = df_1d[df_1d['predition_1d']>0]['label_1d'].mean()
# label_3d = df_3d[df_3d['predition_3d']>0]['label_3d'].mean()
# label_5d = df_5d[df_5d['predition_5d']>0]['label_5d'].mean()
# label_7d = df_7d[df_7d['predition_7d']>0]['label_7d'].mean()

label_7d = df_7d['label_7d'].mean()

print('label_7d: ' + str(label_7d))
print('label_7d: ' + str(label_7d - all_mean['label_7d']))
print('label_7d: ' + str(label_7d - all_code_mean['label_7d']))
