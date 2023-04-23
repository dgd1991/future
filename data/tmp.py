# 一级行业特征
import gc
import os

import numpy as np

from tools.Tools import Tools

import pandas as pd
def func(_df, input_col, output_col):
      l1 = _df[input_col].tolist()
      res = _df[output_col].tolist()
      if np.isnan(res[0]):
         res[0] = 100
      lens = len(l1)
      if lens > 1:
         for i in range(1, lens):
            res[i] = (1 + l1[i]) * res[i - 1]
      _df[output_col] = res
      return _df

is_predict = False
tools = Tools()
k_file_path='E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_2012.csv'
k_file_path_his='E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_2011.csv'
year=2012

raw_k_data = pd.read_csv(k_file_path)
raw_k_data_his = pd.read_csv(k_file_path_his)
raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
raw_k_data = raw_k_data[raw_k_data['industry_id_level3'] > 0]
del raw_k_data_his
gc.collect()
raw_k_data["tradestatus"] = pd.to_numeric(raw_k_data["tradestatus"], errors='coerce')
raw_k_data["turn"] = pd.to_numeric(raw_k_data["turn"], errors='coerce')
raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')
# 需要去除上市当天的股票
raw_k_data = raw_k_data[(raw_k_data['tradestatus'] == 1) & (raw_k_data['turn'] > 0) & (raw_k_data['pctChg'] < 21) & (
			raw_k_data['pctChg'] > -21)]
raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))
raw_k_data['is_new'] = raw_k_data["pctChg"].groupby(level=0).apply(
	lambda x: x.rolling(min_periods=20, window=20, center=False).apply(lambda y: y[0]))
raw_k_data = raw_k_data[raw_k_data['is_new'].map(lambda x: False if np.isnan(x) else True)]
raw_k_data = raw_k_data.reset_index(level=0, drop=True)
raw_k_data = raw_k_data.reset_index(level=0, drop=False)
# raw_k_data['date'].head(5)
raw_k_data["open"] = pd.to_numeric(raw_k_data["open"], errors='coerce')
raw_k_data["close"] = pd.to_numeric(raw_k_data["close"], errors='coerce')
raw_k_data["preclose"] = pd.to_numeric(raw_k_data["preclose"], errors='coerce')
raw_k_data["high"] = pd.to_numeric(raw_k_data["high"], errors='coerce')
raw_k_data["low"] = pd.to_numeric(raw_k_data["low"], errors='coerce')
raw_k_data['date'] = pd.to_datetime(raw_k_data['date'])
raw_k_data["amount"] = pd.to_numeric(raw_k_data["amount"], errors='coerce')
raw_k_data["peTTM"] = pd.to_numeric(raw_k_data["peTTM"], errors='coerce')
raw_k_data["pcfNcfTTM"] = pd.to_numeric(raw_k_data["pcfNcfTTM"], errors='coerce')
raw_k_data["pbMRQ"] = pd.to_numeric(raw_k_data["pbMRQ"], errors='coerce')
raw_k_data["rise"] = raw_k_data["pctChg"].map(lambda x: 1.0 if x > 0 else 0.0)
raw_k_data['pctChg'] = raw_k_data['pctChg'].map(lambda x: x / 100.0)
# 计算的是流通部分的市值
raw_k_data["market_value"] = 0.00000001 * raw_k_data['amount'] / (raw_k_data['turn'].map(lambda x: x / 100.0))
raw_k_data_tmp = raw_k_data[["date", "industry_id_level1", "market_value"]].groupby(
	["industry_id_level1", "date"]).sum()
raw_k_data_tmp.columns = ['industry_id_level1_market_value']
raw_k_data_tmp['industry_id_level1_rise_ratio'] = raw_k_data[["date", "industry_id_level1", "rise"]].groupby(
	["industry_id_level1", "date"]).mean()
raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
raw_k_data = pd.merge(raw_k_data, raw_k_data_tmp, how="left", left_on=["date", "industry_id_level1"],
                      right_on=["date", "industry_id_level1"])
raw_k_data['market_value_ratio'] = raw_k_data["market_value"] / raw_k_data["industry_id_level1_market_value"]

raw_k_data["turn"] = raw_k_data["turn"] * raw_k_data['market_value_ratio']
raw_k_data['open_ratio'] = ((raw_k_data['open'] - raw_k_data['preclose']) / raw_k_data['preclose']) * raw_k_data[
	'market_value_ratio']
raw_k_data['close_ratio'] = ((raw_k_data['close'] - raw_k_data['open']) / raw_k_data['open']) * raw_k_data[
	'market_value_ratio']
raw_k_data['high_ratio'] = ((raw_k_data['high'] - raw_k_data['preclose']) / raw_k_data['preclose']) * raw_k_data[
	'market_value_ratio']
raw_k_data['low_ratio'] = ((raw_k_data['low'] - raw_k_data['preclose']) / raw_k_data['preclose']) * raw_k_data[
	'market_value_ratio']
raw_k_data['pctChg'] = raw_k_data['pctChg'] * raw_k_data['market_value_ratio']
raw_k_data_tmp['industry_id_level1_rise_ratio'] = raw_k_data_tmp['industry_id_level1_rise_ratio'] * raw_k_data[
	'market_value_ratio']

raw_k_data["peTTM"] = raw_k_data["peTTM"] * raw_k_data['market_value_ratio']
raw_k_data["pcfNcfTTM"] = raw_k_data["pcfNcfTTM"] * raw_k_data['market_value_ratio']
raw_k_data["pbMRQ"] = raw_k_data["pbMRQ"] * raw_k_data['market_value_ratio']
raw_k_data["industry_id_level1_rise_ratio"] = raw_k_data["industry_id_level1_rise_ratio"] * raw_k_data[
	'market_value_ratio']

# 新增板块的最近n天的涨停股票数量，主要评估板块热度
raw_k_data['pctChg_limit'] = raw_k_data[['pctChg', 'turn', 'close', 'high', 'low', 'isST']].apply(
	lambda x: tools.code_pctChg_limit_type(x.pctChg, x.isST, x.high, x.low, x.close), axis=1)
raw_k_data['pctChg_up_limit'] = raw_k_data['pctChg_limit'].map(lambda x: 1 if x == 1 else 0)
raw_k_data['pctChg_down_limit'] = raw_k_data['pctChg_limit'].map(lambda x: 1 if x == 2 else 0)

industry_id_level1_k_data = raw_k_data[
	["industry_id_level1", "open_ratio", "close_ratio", "high_ratio", "low_ratio", "turn", "date", "pctChg", "peTTM",
	 "pcfNcfTTM", "pbMRQ", 'industry_id_level1_rise_ratio', 'market_value', 'pctChg_up_limit',
	 'pctChg_down_limit']].groupby(['industry_id_level1', 'date']).sum().round(5)
industry_id_level1_k_data.columns = ["industry_id_level1_open_ratio", "industry_id_level1_close_ratio",
                                     "industry_id_level1_high_ratio", "industry_id_level1_low_ratio",
                                     "industry_id_level1_turn", "industry_id_level1_pctChg", "industry_id_level1_peTTM",
                                     "industry_id_level1_pcfNcfTTM", "industry_id_level1_pbMRQ",
                                     'industry_id_level1_rise_ratio', 'industry_id_level1_market_value',
                                     'industry_id_level1_pctChg_up_limit', 'industry_id_level1_pctChg_down_limit']

del raw_k_data
gc.collect()
print(industry_id_level1_k_data[["industry_id_level1_close_ratio"]].head(5))
if os.path.isfile('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level1_' + str(year - 1) + '.csv'):
	industry_id_level1_k_data_his = pd.read_csv(
		'E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level1_' + str(year - 1) + '.csv')
	industry_id_level1_k_data_his = industry_id_level1_k_data_his[
		['industry_id_level1', 'date', 'industry_id_level1_close']]
	industry_id_level1_k_data_his['date'] = pd.to_datetime(industry_id_level1_k_data_his['date'])
industry_id_level1_k_data = industry_id_level1_k_data.reset_index(level=0, drop=False)
industry_id_level1_k_data = industry_id_level1_k_data.reset_index(level=0, drop=False)
print(industry_id_level1_k_data[['date',"industry_id_level1_close_ratio"]].head(5))
if os.path.isfile('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level1_' + str(year - 1) + '.csv'):
	industry_id_level1_k_data = pd.merge(industry_id_level1_k_data, industry_id_level1_k_data_his, how="left",
	                                     left_on=['industry_id_level1', 'date'],
	                                     right_on=['industry_id_level1', 'date'])
else:
	industry_id_level1_k_data['industry_id_level1_close'] = industry_id_level1_k_data[
		'industry_id_level1_open_ratio'].apply(lambda x: 1000)


print(industry_id_level1_k_data[['date',"industry_id_level1_close_ratio"]].head(5))

industry_id_level1_k_data = industry_id_level1_k_data.groupby('industry_id_level1').apply(lambda x: x.set_index('date', drop=True))

industry_id_level1_k_data = industry_id_level1_k_data.groupby(level=0).apply(lambda x: func(x, 'industry_id_level1_pctChg', 'industry_id_level1_close')).round(5)
industry_id_level1_k_data['industry_id_level1_preclose'] = industry_id_level1_k_data['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0])).round(5)
industry_id_level1_k_data['industry_id_level1_open'] = industry_id_level1_k_data['industry_id_level1_preclose']*(industry_id_level1_k_data['industry_id_level1_open_ratio'].apply(lambda x: x + 1)).round(5)
industry_id_level1_k_data['industry_id_level1_high'] = industry_id_level1_k_data['industry_id_level1_preclose']*(industry_id_level1_k_data['industry_id_level1_high_ratio'].apply(lambda x: x + 1)).round(5)
industry_id_level1_k_data['industry_id_level1_low'] = industry_id_level1_k_data['industry_id_level1_preclose']*(industry_id_level1_k_data['industry_id_level1_low_ratio'].apply(lambda x: x + 1)).round(5)

# 新增板块的最近n天的涨停股票数量，主要评估板块热度
for day_cnt in [3, 7, 15, 30]:
	industry_id_level1_k_data['industry_id_level1_pctChg_up_limit_' + str(day_cnt)] = industry_id_level1_k_data['industry_id_level1_pctChg_up_limit'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=day_cnt, center=False).sum())
	industry_id_level1_k_data['industry_id_level1_pctChg_down_limit_' + str(day_cnt)] = industry_id_level1_k_data['industry_id_level1_pctChg_down_limit'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=day_cnt, center=False).sum())

# 写出指数点数
industry_id_level1_k_data_out = industry_id_level1_k_data[['industry_id_level1_open', 'industry_id_level1_close', 'industry_id_level1_high', 'industry_id_level1_low']]
industry_id_level1_k_data_out = industry_id_level1_k_data_out.reset_index(level=0, drop=False)
industry_id_level1_k_data_out = industry_id_level1_k_data_out.reset_index(level=0, drop=False)
print(industry_id_level1_k_data_out[['date',"industry_id_level1_open"]].head(5))
