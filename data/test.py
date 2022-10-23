import pandas as pd
year = 2020
model_name = 'model_v4'
# test_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name=model_name, year=str(year))
# df = pd.read_csv(test_data)
# # df = df[df['date']==20200126]
# test_data1 = 'E:/pythonProject/future/data/datafile/prediction_result/{model_name}/prediction_result_{year}.csv'.format(model_name=model_name,year=str(year))
# df1 = pd.read_csv(test_data1)
# df1.columns = ['date', 'code', 'predition', 'label_7', 'label_7_real', 'label_7_weight', 'label_7_max', 'label_7_max_real', 'label_7_max_weight', 'label_15', 'label_15_real', 'label_15_weight', 'label_15_max', 'label_15_max_real', 'label_15_max_weight']
# # df1 = df1[df1['date']==20200126]
# # print(df.shape)
# # print(df1.shape)
# df = df.groupby('date')['code'].count()
# df1 = df1.groupby('date')['code'].count()
# print(df)
# print(df1)
import os
test_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name='tmp', year=str(2021))
if os.path.isfile(test_data):
	os.remove(test_data)
df = pd.read_csv(test_data)
df = df[df['date']==20200120]
print(df.shape)
# df.to_csv('E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}_test.csv'.format(model_name=model_name,year=str(year)), mode='a', header=True, index=False, encoding='utf-8')


raw_k_data["tradestatus"] = pd.to_numeric(raw_k_data["tradestatus"], errors='coerce')
raw_k_data["turn"] = pd.to_numeric(raw_k_data["turn"], errors='coerce')
raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')
raw_k_data = raw_k_data[(raw_k_data['tradestatus'] == 1) & (raw_k_data['turn'] > 0) & (raw_k_data['pctChg'] <= 20) & (raw_k_data['pctChg'] >= -20)]
raw_k_data["open"] = pd.to_numeric(raw_k_data["open"], errors='coerce')
raw_k_data["close"] = pd.to_numeric(raw_k_data["close"], errors='coerce')
raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')
raw_k_data["preclose"] = pd.to_numeric(raw_k_data["preclose"], errors='coerce')
raw_k_data["high"] = pd.to_numeric(raw_k_data["high"], errors='coerce')
raw_k_data["low"] = pd.to_numeric(raw_k_data["low"], errors='coerce')
raw_k_data['date'] = pd.to_datetime(raw_k_data['date'])
raw_k_data['open_ratio'] = ((raw_k_data['open'] - raw_k_data['preclose']) / raw_k_data['preclose'])
raw_k_data['close_ratio'] = ((raw_k_data['close'] - raw_k_data['open']) / raw_k_data['open'])
raw_k_data['high_ratio'] = ((raw_k_data['high'] - raw_k_data['preclose']) / raw_k_data['preclose'])
raw_k_data['low_ratio'] = ((raw_k_data['low'] - raw_k_data['preclose']) / raw_k_data['preclose'])
raw_k_data['amount'] = raw_k_data['amount']
raw_k_data['pctChg'] = raw_k_data['pctChg']
raw_k_data['peTTM'] = raw_k_data['peTTM']
raw_k_data['pcfNcfTTM'] = raw_k_data['pcfNcfTTM']
raw_k_data['pbMRQ'] = raw_k_data['pbMRQ']
raw_k_data['isST'] = raw_k_data['isST']


