import copy

import pandas as pd
# 省略问题修改配置, 打印100列数据
from feature.feature_process import float2Bucket

pd.set_option('display.max_columns', 200)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)

import numpy as np
# quater_path = 'E:/pythonProject/future/data/datafile/code_quarter_data_v2_all.csv'
# file = pd.read_csv(quater_path, encoding='utf-8', dtype={'industry': str})
# # print(file.dtypes)
# file = file["industry"].drop_duplicates()
# # print(type(file))
# index = 1
# dict = {}
# for name in file:
# 	if type(name) == type(np.nan):
# 		dict[name] = 0
# 	else:
# 		dict[name] = index
# 		index += 1
# dict["max"] = index - 1
# np.save('E:/pythonProject/future/common/industry_dict.npy', dict)  # 注意带上后缀名
#
# # Load
# load_dict = np.load('E:/pythonProject/future/common/industry_dict.npy').item()
# print(load_dict)

file_name = 'E:/pythonProject/future/data/datafile/code_k_data_v2_2007.csv'
raw_k_data = pd.read_csv(file_name, encoding='utf-8')
raw_k_data["tradestatus"] = pd.to_numeric(raw_k_data["tradestatus"], errors='coerce')
raw_k_data["turn"] = pd.to_numeric(raw_k_data["turn"], errors='coerce')
raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')
raw_k_data = raw_k_data[(raw_k_data['tradestatus'] == 1) & (raw_k_data['turn'] > 0) & (raw_k_data['pctChg'] <= 20) & (raw_k_data['pctChg'] >= -20)]
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
raw_k_data["rise"] = raw_k_data["pctChg"].map(lambda x: 1.0 if x>0 else 0.0)
raw_k_data["volume_total"] = pd.to_numeric(raw_k_data["volume"], errors='coerce')/1000000/raw_k_data["turn"]
raw_k_data_tmp = raw_k_data[["date","industry", "volume_total"]].groupby(["date","industry"]).sum()
raw_k_data_tmp.columns = ['industry_volume_total']
raw_k_data_tmp['rise_ratio'] = raw_k_data[["date","industry", "rise"]].groupby(["date","industry"]).mean()
raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
raw_k_data = pd.merge(raw_k_data, raw_k_data_tmp, how="left", left_on=["date","industry"], right_on=["date","industry"])
raw_k_data['volume_total_ratio'] = raw_k_data["volume_total"]/raw_k_data["industry_volume_total"]

raw_k_data["open"] = raw_k_data["open"]*raw_k_data['volume_total_ratio']
raw_k_data["close"] = raw_k_data["close"]*raw_k_data['volume_total_ratio']
raw_k_data["preclose"] = raw_k_data["preclose"]*raw_k_data['volume_total_ratio']
raw_k_data["high"] = raw_k_data["high"]*raw_k_data['volume_total_ratio']
raw_k_data["low"] = raw_k_data["low"]*raw_k_data['volume_total_ratio']
raw_k_data["turn"] = raw_k_data["turn"]*raw_k_data['volume_total_ratio']
raw_k_data["amount"] = raw_k_data["amount"]*raw_k_data['volume_total_ratio']
raw_k_data["pctChg"] = raw_k_data["pctChg"]*raw_k_data['volume_total_ratio']
raw_k_data["peTTM"] = raw_k_data["peTTM"]*raw_k_data['volume_total_ratio']
raw_k_data["pcfNcfTTM"] = raw_k_data["pcfNcfTTM"]*raw_k_data['volume_total_ratio']
raw_k_data["pbMRQ"] = raw_k_data["pbMRQ"]*raw_k_data['volume_total_ratio']
raw_k_data["rise_ratio"] = raw_k_data["rise_ratio"]*raw_k_data['volume_total_ratio']

# raw_k_data.set_index(['date',"industry"],inplace=True)
# raw_k_data["code_ratio"] = raw_k_data["volume_total"]/(raw_k_data["volume_total"].sum(level = [["date", "industry"]]))
# print(raw_k_data[raw_k_data['industry'] == 28][["date","code","industry","volume", "turn","volume_total","industry_volume_total", "pctChg", 'rise_ratio']])


industry_k_data = raw_k_data[["industry","open","close","preclose","high","low","turn","date","amount","pctChg","peTTM","pcfNcfTTM","pbMRQ", 'rise_ratio']].groupby(['industry','date']).sum()

industry_k_data['industry_open_ratio'] = ((industry_k_data['open'] - industry_k_data['preclose']) / industry_k_data['preclose'])
industry_k_data['industry_close_ratio'] = ((industry_k_data['close'] - industry_k_data['open']) / industry_k_data['open'])
industry_k_data['industry_high_ratio'] = ((industry_k_data['high'] - industry_k_data['preclose']) / industry_k_data['preclose'])
industry_k_data['industry_low_ratio'] = ((industry_k_data['low'] - industry_k_data['preclose']) / industry_k_data['preclose'])

industry_k_data['industry_open_ratio_7d_avg'] = industry_k_data['industry_open_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
industry_k_data['industry_close_ratio_7d_avg'] = industry_k_data['industry_close_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
industry_k_data['industry_high_ratio_7d_avg'] = industry_k_data['industry_high_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
industry_k_data['industry_low_ratio_7d_avg'] = industry_k_data['industry_low_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())

industry_k_data['industry_amount'] = industry_k_data['amount'].map(lambda x: None if x == '' else float2Bucket(float(x) * 0.00000005, 1, 0, 40000, 40000))
industry_k_data['industry_peTTM'] = industry_k_data['peTTM'].map(lambda x: float2Bucket(float(x), 2, 0, 2000, 4000))
industry_k_data['industry_pcfNcfTTM'] = industry_k_data['pcfNcfTTM'].map(lambda x: float2Bucket(float(x), 10, 0, 100, 1000))
industry_k_data['industry_pbMRQ'] = industry_k_data['pbMRQ'].map(lambda x: float2Bucket(float(x), 10, 0, 500, 5000))
industry_k_data['industry_turn'] = industry_k_data['turn'].map(lambda x: float2Bucket(float(x), 2000, 0, 1, 2000))

# kdj 5, 9, 19, 36, 45, 73，
# 任意初始化，超过30天后的kdj值基本一样
tmp = industry_k_data[['low', 'high', 'close']]
for day_cnt in [5, 9, 19, 73]:
	# industry_k_data['industry_rsv'] = industry_k_data[['low', 'high', 'close']].groupby(level=0).apply(lambda x: (x.close-x.low.rolling(min_periods=1, window=day_cnt, center=False).min())/(x.high.rolling(min_periods=1, window=day_cnt, center=False).max()-x.low.rolling(min_periods=1, window=day_cnt, center=False).min()))
	tmp['min'] = tmp['low'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
	tmp['max'] = tmp['high'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
	industry_k_data['industry_rsv_' + str(day_cnt)] = (tmp['close'] - tmp['min'])/(tmp['max'] - tmp['min'])
	industry_k_data['industry_k_value_' + str(day_cnt)] = industry_k_data['industry_rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
	industry_k_data['industry_d_value_' + str(day_cnt)] = industry_k_data['industry_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
	industry_k_data['industry_j_value_' + str(day_cnt)] = 3 * industry_k_data['industry_k_value_' + str(day_cnt)] - 2 * industry_k_data['industry_d_value_' + str(day_cnt)]
	industry_k_data['industry_k_value_trend_' + str(day_cnt)] = industry_k_data['industry_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1])-y[0]))
	industry_k_data['industry_kd_value' + str(day_cnt)] = (industry_k_data['industry_k_value_' + str(day_cnt)] - industry_k_data['industry_d_value_' + str(day_cnt)]).rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1))
# macd 12
# 任意初始化，超过30天后的macd值基本一样
tmp = industry_k_data[['close']]
tmp['ema_12'] = tmp['close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 13, adjust=False).mean())
tmp['ema_26'] = tmp['close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 27, adjust=False).mean())
tmp['industry_macd_dif'] = tmp['ema_12'] - tmp['ema_26']
tmp['industry_macd_dea'] = tmp['industry_macd_dif'].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0 / 5, adjust=False).mean())
industry_k_data['industry_macd'] = (tmp['industry_macd_dif'] - tmp['industry_macd_dea']) * 2

# boll线
tmp['mb_20'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).mean())
tmp['md_20'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).std())
tmp['up_20'] = tmp['mb_20'] + 2 * tmp['md_20']
tmp['dn_20'] = tmp['mb_20'] - 2 * tmp['md_20']
industry_k_data['industry_width_20'] = 4 * tmp['md_20'] / tmp['mb_20']
industry_k_data['industry_close_mb20_diff'] = (tmp['close'] - tmp['mb_20'])/(2 * tmp['md_20'])

# cr指标
tmp = industry_k_data[['close', 'open', 'high', 'low']]
tmp['cr_m'] = (tmp['close'] + tmp['open'] + tmp['high'] + tmp['low'])/4
tmp['cr_ym'] = tmp['cr_m'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0]))
tmp['cr_p1_day'] = (tmp['high'] - tmp['cr_ym'])
tmp['cr_p2_day'] = (tmp['cr_ym'] - tmp['low'])
for day_cnt in (3, 5, 10, 20, 40):
	tmp['cr_p1_' + str(day_cnt) + 'd'] = tmp['cr_p1_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
	tmp['cr_p2_' + str(day_cnt) + 'd'] = tmp['cr_p2_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
	industry_k_data['industry_cr_' + str(day_cnt) + 'd'] = tmp['cr_p1_' + str(day_cnt) + 'd'] / tmp['cr_p2_' + str(day_cnt) + 'd']
# rsi指标
tmp = industry_k_data[['close', 'preclose']]
tmp['price_dif'] = tmp['close'] - tmp['preclose']
tmp['rsi_positive'] = tmp['price_dif'].apply(lambda x: max(x, 0))
tmp['rsi_all'] = tmp['price_dif'].apply(lambda x: abs(x))
for day_cnt in (3, 5, 10, 20, 40):
	tmp['rsi_positive_sum'] = tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
	tmp['rsi_all_sum'] = tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
	industry_k_data['industry_rsi_' + str(day_cnt) + 'd'] =  tmp['rsi_positive_sum'] / tmp['rsi_all_sum']

tmp = industry_k_data[['turn', 'close']]
day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
for index in range(len(day_cnt_list)):
	day_cnt = day_cnt_list[index]
	day_cnt_last = day_cnt_list[index-1]
	industry_k_data['industry_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_k_data['turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
	industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_k_data['turn']/industry_k_data['industry_turn_' + str(day_cnt) + 'd' + '_avg']
	industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
	industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
	if index>0:
		industry_k_data['industry_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_k_data['industry_turn_' + str(day_cnt_last) + 'd' + '_avg']/industry_k_data['industry_turn_' + str(day_cnt) + 'd' + '_avg']

	tmp['industry_close_' + str(day_cnt) + 'd' + '_avg'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
	industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_k_data['close']/tmp['industry_close_' + str(day_cnt) + 'd' + '_avg']
	industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_k_data['close']/industry_k_data['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
	industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_k_data['close']/industry_k_data['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
	industry_k_data['industry_close_' + str(day_cnt) + 'd' + '_dif'] = industry_k_data['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 1.0*(y[day_cnt-1])/y[0]))
	if index>0:
		industry_k_data['industry_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_close_' + str(day_cnt_last) + 'd' + '_avg']/tmp['industry_close_' + str(day_cnt) + 'd' + '_avg']

for index in range(len(day_cnt_list)):
	day_cnt = day_cnt_list[index]
	day_cnt_last = day_cnt_list[index-1]
	if index > 0:
		industry_k_data['industry_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_k_data['industry_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 20, 0, 10, 200))
		industry_k_data['industry_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_k_data['industry_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 4, 200))
	industry_k_data['industry_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_k_data['industry_turn_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 2, 0, 100, 200))
	industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
	industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 10, 1, 50, 500))
	industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

	industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
	industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 100, 0, 1, 100))
	industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 1, 10, 500))
	industry_k_data['industry_close_' + str(day_cnt) + 'd' + '_dif'] = industry_k_data['industry_close_' + str(day_cnt) + 'd' + '_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

industry_k_data = industry_k_data.reset_index(level=0, drop=False)
industry_k_data = industry_k_data.reset_index(level=0, drop=False)
print(industry_k_data)