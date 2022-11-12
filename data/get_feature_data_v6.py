import copy
import gc
import os
from itertools import zip_longest

import pandas as pd

from feature.feature_process import *
from tools.Tools import Tools
from feature.get_technical_indicators import *

pd.set_option('display.max_columns', 200)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)


class Feature(object):
	def __init__(self,  k_file_path, year, quarter_file_path):
		self.k_file_path = k_file_path + str(year) + '.csv'
		self.k_file_path_his = k_file_path + str(year - 1) + '.csv'
		self.quarter_file_path = quarter_file_path
		self.year = year
		self.tools = Tools()

	def feature_process(self):
		raw_k_data = pd.read_csv(self.k_file_path)
		raw_k_data_his = pd.read_csv(self.k_file_path_his)
		raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
		del raw_k_data_his
		gc.collect()
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
		raw_k_data['pctChg'] = raw_k_data['pctChg'].map(lambda x: x/100.0)
		raw_k_data['code_market'] = raw_k_data['code'].map(lambda x: self.tools.code_market(x))

		raw_k_data['peTTM'] = raw_k_data['peTTM']
		raw_k_data['pcfNcfTTM'] = raw_k_data['pcfNcfTTM']
		raw_k_data['pbMRQ'] = raw_k_data['pbMRQ']
		raw_k_data['isST'] = raw_k_data['isST']

		raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))

		feature_all = copy.deepcopy(raw_k_data[['industry_name_level1','industry_name_level2','industry_name_level3','industry_id_level1','industry_id_level2','industry_id_level3','open_ratio','close_ratio','high_ratio','low_ratio','pctChg','code_market']])
		feature_all['open_ratio_7d_avg'] = raw_k_data.groupby(level=0)['open_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		feature_all['close_ratio_7d_avg'] = raw_k_data.groupby(level=0)['close_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		feature_all['high_ratio_7d_avg'] = raw_k_data.groupby(level=0)['high_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		feature_all['low_ratio_7d_avg'] = raw_k_data.groupby(level=0)['low_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())

		feature_all['amount'] = raw_k_data['amount'].map(lambda x: None if x == '' else float2Bucket(float(x) * 0.00000001, 1, 0, 10000, 10000))
		feature_all['peTTM'] = raw_k_data['peTTM'].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 1500, 1500))
		feature_all['pcfNcfTTM'] = raw_k_data['pcfNcfTTM'].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 2000, 2000))
		feature_all['pbMRQ'] = raw_k_data['pbMRQ'].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 2000, 2000))
		feature_all['isST'] = raw_k_data['isST'].map(lambda x: float2Bucket(float(x), 1, 0, 3, 3))

		feature_all['turn'] = raw_k_data['turn'].map(lambda x: float2Bucket(float(x), 1, 0, 200, 200))

		# kdj 5, 9, 19, 36, 45, 73，
		# 任意初始化，超过30天后的kdj值基本一样
		tmp = raw_k_data[['low', 'high', 'close']]
		for day_cnt in [5, 9, 19, 73]:
			tmp['min'] = tmp['low'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			tmp['max'] = tmp['high'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			feature_all['rsv_' + str(day_cnt)] = (tmp['close'] - tmp['min'])/(tmp['max'] - tmp['min'])
			feature_all['k_value_' + str(day_cnt)] = feature_all['rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
			feature_all['d_value_' + str(day_cnt)] = feature_all['k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
			feature_all['j_value_' + str(day_cnt)] = 3 * feature_all['k_value_' + str(day_cnt)] - 2 * feature_all['d_value_' + str(day_cnt)]
			feature_all['k_value_trend_' + str(day_cnt)] = feature_all['k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1])-y[0]))
			feature_all['kd_value' + str(day_cnt)] = (feature_all['k_value_' + str(day_cnt)] - feature_all['d_value_' + str(day_cnt)]).rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1))

		# macd 12
		# 任意初始化，超过30天后的macd值基本一样
		tmp = raw_k_data[['close']]
		tmp['ema_12'] = tmp['close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 13, adjust=False).mean())
		tmp['ema_26'] = tmp['close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 27, adjust=False).mean())
		tmp['macd_dif'] = tmp['ema_12'] - tmp['ema_26']
		tmp['macd_dea'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0 / 5, adjust=False).mean())
		tmp['macd_dif_max'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
		tmp['macd_dif_min'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
		tmp['macd_dea_max'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
		tmp['macd_dea_min'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
		tmp['macd'] = (tmp['macd_dif'] - tmp['macd_dea']) * 2
		feature_all['macd_positive'] = tmp['macd'].apply(lambda x: 1 if x>0 else 0)
		feature_all['macd_dif_ratio'] = (tmp['macd_dif']-tmp['macd_dif_min'])/(tmp['macd_dif_max']-tmp['macd_dif_min'])
		for day_cnt in [2, 3, 5, 10, 20, 40]:
			feature_all['macd_dif_' + str(day_cnt)] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			feature_all['macd_dea_' + str(day_cnt)] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			feature_all['macd_' + str(day_cnt)] = tmp['macd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			feature_all['macd_positive_ratio_' + str(day_cnt)] = feature_all['macd_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
		feature_all['macd_dif_dea'] = (tmp['macd_dif']-tmp['macd_dea']).groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1)))

		# boll线
		tmp['mb_20'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).mean())
		tmp['md_20'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).std())
		tmp['up_20'] = tmp['mb_20'] + 2 * tmp['md_20']
		tmp['dn_20'] = tmp['mb_20'] - 2 * tmp['md_20']
		feature_all['width_20'] = 4 * tmp['md_20'] / tmp['mb_20']
		feature_all['close_mb20_diff'] = (tmp['close'] - tmp['mb_20'])/(2 * tmp['md_20'])

		# cr指标
		tmp = raw_k_data[['close', 'open', 'high', 'low']]
		tmp['cr_m'] = (tmp['close'] + tmp['open'] + tmp['high'] + tmp['low'])/4
		tmp['cr_ym'] = tmp['cr_m'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0]))
		tmp['cr_p1_day'] = (tmp['high'] - tmp['cr_ym'])
		tmp['cr_p2_day'] = (tmp['cr_ym'] - tmp['low'])
		for day_cnt in (3, 5, 10, 20, 40):
			tmp['cr_p1_' + str(day_cnt) + 'd'] = tmp['cr_p1_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			tmp['cr_p2_' + str(day_cnt) + 'd'] = tmp['cr_p2_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			feature_all['cr_' + str(day_cnt) + 'd'] = tmp['cr_p1_' + str(day_cnt) + 'd'] / tmp['cr_p2_' + str(day_cnt) + 'd']
			feature_all['cr_' + str(day_cnt) + 'd'] = feature_all['cr_' + str(day_cnt) + 'd'].map(lambda x: float2Bucket(float(x)*100, 0.2, 0, 500, 100))
		# rsi指标
		tmp = raw_k_data[['close', 'preclose']]
		tmp['price_dif'] = tmp['close'] - tmp['preclose']
		tmp['rsi_positive'] = tmp['price_dif'].apply(lambda x: max(x, 0))
		tmp['rsi_all'] = tmp['price_dif'].apply(lambda x: abs(x))
		for day_cnt in (3, 5, 10, 20, 40):
			feature_all['rsi_' + str(day_cnt) + 'd'] = tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum()) / tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())

		day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
		for index in range(len(day_cnt_list)):
			day_cnt = day_cnt_list[index]
			day_cnt_last = day_cnt_list[index-1]
			feature_all['turn_' + str(day_cnt) + 'd' + '_avg'] = raw_k_data.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'] = raw_k_data['turn']/feature_all['turn_' + str(day_cnt) + 'd' + '_avg']
			feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			if index>0:
				feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = feature_all['turn_' + str(day_cnt_last) + 'd' + '_avg']/feature_all['turn_' + str(day_cnt) + 'd' + '_avg']

			raw_k_data['close_' + str(day_cnt) + 'd' + '_avg'] = raw_k_data.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			feature_all['close_' + str(day_cnt) + 'd' + 'avg_dif'] = raw_k_data['close']/raw_k_data['close_' + str(day_cnt) + 'd' + '_avg']
			feature_all['close_' + str(day_cnt) + 'd' + 'max_dif'] = raw_k_data['close']/raw_k_data.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			feature_all['close_' + str(day_cnt) + 'd' + 'min_dif'] = raw_k_data['close']/raw_k_data.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			feature_all['close_' + str(day_cnt) + 'd' + '_dif'] = raw_k_data.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 1.0*(y[day_cnt-1])/y[0]))
			if index>0:
				feature_all['close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = raw_k_data['close_' + str(day_cnt_last) + 'd' + '_avg']/raw_k_data['close_' + str(day_cnt) + 'd' + '_avg']

		for index in range(len(day_cnt_list)):
			day_cnt = day_cnt_list[index]
			day_cnt_last = day_cnt_list[index - 1]
			if index > 0:
				feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 20, 0, 10, 200))
				feature_all['close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = feature_all['close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 4, 200))
			feature_all['turn_' + str(day_cnt) + 'd' + '_avg'] = feature_all['turn_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 2, 0, 100, 200))
			feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
			feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 10, 1, 50, 500))
			feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

			feature_all['close_' + str(day_cnt) + 'd' + 'avg_dif'] = feature_all['close_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
			feature_all['close_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['close_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 100, 0, 1, 100))
			feature_all['close_' + str(day_cnt) + 'd' + 'min_dif'] = feature_all['close_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 1, 10, 500))
			feature_all['close_' + str(day_cnt) + 'd' + '_dif'] = feature_all['close_' + str(day_cnt) + 'd' + '_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

		feature_all = feature_all.reset_index(level=0, drop=False)
		feature_all = feature_all.reset_index(level=0, drop=False)
		feature_all = feature_all.sort_values(['date', 'code'])
		feature_all = feature_all[feature_all['date'] > str(self.year)]

		# 一级行业特征
		raw_k_data = pd.read_csv(self.k_file_path)
		raw_k_data_his = pd.read_csv(self.k_file_path_his)
		raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
		del raw_k_data_his
		gc.collect()
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
		raw_k_data_tmp = raw_k_data[["date","industry_id_level1", "volume_total"]].groupby(["date","industry_id_level1"]).sum()
		raw_k_data_tmp.columns = ['industry_id_level1_volume_total']
		raw_k_data_tmp['industry_id_level1_rise_ratio'] = raw_k_data[["date","industry_id_level1", "rise"]].groupby(["date","industry_id_level1"]).mean()
		raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
		raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
		raw_k_data = pd.merge(raw_k_data, raw_k_data_tmp, how="left", left_on=["date","industry_id_level1"], right_on=["date","industry_id_level1"])
		raw_k_data['volume_total_ratio'] = raw_k_data["volume_total"]/raw_k_data["industry_id_level1_volume_total"]

		raw_k_data["open"] = raw_k_data["open"]*raw_k_data['volume_total_ratio']
		raw_k_data["close"] = raw_k_data["close"]*raw_k_data['volume_total_ratio']
		raw_k_data["preclose"] = raw_k_data["preclose"]*raw_k_data['volume_total_ratio']
		raw_k_data["high"] = raw_k_data["high"]*raw_k_data['volume_total_ratio']
		raw_k_data["low"] = raw_k_data["low"]*raw_k_data['volume_total_ratio']
		raw_k_data["turn"] = raw_k_data["turn"]*raw_k_data['volume_total_ratio']
		raw_k_data["amount"] = raw_k_data["amount"]*raw_k_data['volume_total_ratio']
		raw_k_data['pctChg'] = raw_k_data['pctChg'].map(lambda x: x/100.0)
		raw_k_data["pctChg"] = raw_k_data["pctChg"]*raw_k_data['volume_total_ratio']
		raw_k_data["peTTM"] = raw_k_data["peTTM"]*raw_k_data['volume_total_ratio']
		raw_k_data["pcfNcfTTM"] = raw_k_data["pcfNcfTTM"]*raw_k_data['volume_total_ratio']
		raw_k_data["pbMRQ"] = raw_k_data["pbMRQ"]*raw_k_data['volume_total_ratio']
		raw_k_data["industry_id_level1_rise_ratio"] = raw_k_data["industry_id_level1_rise_ratio"]*raw_k_data['volume_total_ratio']

		industry_id_level1_k_data = raw_k_data[["industry_id_level1","open","close","preclose","high","low","turn","date","amount","pctChg","peTTM","pcfNcfTTM","pbMRQ", 'industry_id_level1_rise_ratio']].groupby(['industry_id_level1','date']).sum()
		industry_id_level1_k_data.columns = ["industry_id_level1_open","industry_id_level1_close","industry_id_level1_preclose","industry_id_level1_high","industry_id_level1_low","industry_id_level1_turn","industry_id_level1_amount","industry_id_level1_pctChg","industry_id_level1_peTTM","industry_id_level1_pcfNcfTTM","industry_id_level1_pbMRQ", 'industry_id_level1_rise_ratio']

		industry_id_level1_k_data['industry_id_level1_open_ratio'] = ((industry_id_level1_k_data["industry_id_level1_open"] - industry_id_level1_k_data["industry_id_level1_preclose"]) / industry_id_level1_k_data["industry_id_level1_preclose"])
		industry_id_level1_k_data['industry_id_level1_close_ratio'] = ((industry_id_level1_k_data["industry_id_level1_close"] - industry_id_level1_k_data["industry_id_level1_open"]) / industry_id_level1_k_data["industry_id_level1_open"])
		industry_id_level1_k_data['industry_id_level1_high_ratio'] = ((industry_id_level1_k_data["industry_id_level1_high"] - industry_id_level1_k_data["industry_id_level1_preclose"]) / industry_id_level1_k_data["industry_id_level1_preclose"])
		industry_id_level1_k_data['industry_id_level1_low_ratio'] = ((industry_id_level1_k_data["industry_id_level1_low"] - industry_id_level1_k_data["industry_id_level1_preclose"]) / industry_id_level1_k_data["industry_id_level1_preclose"])

		industry_id_level1_k_data['industry_id_level1_open_ratio_7d_avg'] = industry_id_level1_k_data['industry_id_level1_open_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_id_level1_k_data['industry_id_level1_close_ratio_7d_avg'] = industry_id_level1_k_data['industry_id_level1_close_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_id_level1_k_data['industry_id_level1_high_ratio_7d_avg'] = industry_id_level1_k_data['industry_id_level1_high_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_id_level1_k_data['industry_id_level1_low_ratio_7d_avg'] = industry_id_level1_k_data['industry_id_level1_low_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())

		industry_id_level1_k_data['industry_id_level1_turn'] = industry_id_level1_k_data["industry_id_level1_turn"].map(lambda x: float2Bucket(float(x), 1, 0, 200, 200))
		industry_id_level1_k_data['industry_id_level1_amount'] = industry_id_level1_k_data["industry_id_level1_amount"].map(lambda x: None if x == '' else float2Bucket(float(x) * 0.00000001, 1, 0, 10000, 10000))
		industry_id_level1_k_data['industry_id_level1_peTTM'] = industry_id_level1_k_data["industry_id_level1_peTTM"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 1500, 1500))
		industry_id_level1_k_data['industry_id_level1_pcfNcfTTM'] = industry_id_level1_k_data["industry_id_level1_pcfNcfTTM"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 2000, 2000))
		industry_id_level1_k_data['industry_id_level1_pbMRQ'] = industry_id_level1_k_data["industry_id_level1_pbMRQ"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 2000, 2000))


		# kdj 5, 9, 19, 36, 45, 73，
		# 任意初始化，超过30天后的kdj值基本一样
		tmp = industry_id_level1_k_data[["industry_id_level1_low", "industry_id_level1_high", "industry_id_level1_close"]]
		for day_cnt in [5, 9, 19, 73]:
		# industry_id_level1_k_data['industry_id_level1_rsv'] = industry_id_level1_k_data[["industry_id_level1_low", "industry_id_level1_high", "industry_id_level1_close"]].groupby(level=0).apply(lambda x: (x.close-x.low.rolling(min_periods=1, window=day_cnt, center=False).min())/(x.high.rolling(min_periods=1, window=day_cnt, center=False).max()-x.low.rolling(min_periods=1, window=day_cnt, center=False).min()))
			tmp['min'] = tmp["industry_id_level1_low"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			tmp['max'] = tmp["industry_id_level1_high"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_id_level1_k_data['industry_id_level1_rsv_' + str(day_cnt)] = (tmp["industry_id_level1_close"] - tmp['min'])/(tmp['max'] - tmp['min'])
			industry_id_level1_k_data['industry_id_level1_k_value_' + str(day_cnt)] = industry_id_level1_k_data['industry_id_level1_rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
			industry_id_level1_k_data['industry_id_level1_d_value_' + str(day_cnt)] = industry_id_level1_k_data['industry_id_level1_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
			industry_id_level1_k_data['industry_id_level1_j_value_' + str(day_cnt)] = 3 * industry_id_level1_k_data['industry_id_level1_k_value_' + str(day_cnt)] - 2 * industry_id_level1_k_data['industry_id_level1_d_value_' + str(day_cnt)]
			industry_id_level1_k_data['industry_id_level1_k_value_trend_' + str(day_cnt)] = industry_id_level1_k_data['industry_id_level1_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1])-y[0]))
			industry_id_level1_k_data['industry_id_level1_kd_value' + str(day_cnt)] = (industry_id_level1_k_data['industry_id_level1_k_value_' + str(day_cnt)] - industry_id_level1_k_data['industry_id_level1_d_value_' + str(day_cnt)]).rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1))

		# macd 12
		# 任意初始化，超过30天后的macd值基本一样
		tmp = industry_id_level1_k_data[['industry_id_level1_close']]
		tmp['ema_12'] = tmp['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 13, adjust=False).mean())
		tmp['ema_26'] = tmp['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 27, adjust=False).mean())
		tmp['macd_dif'] = tmp['ema_12'] - tmp['ema_26']
		tmp['macd_dea'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0 / 5, adjust=False).mean())
		tmp['macd_dif_max'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
		tmp['macd_dif_min'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
		tmp['macd_dea_max'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
		tmp['macd_dea_min'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
		tmp['macd'] = (tmp['macd_dif'] - tmp['macd_dea']) * 2
		industry_id_level1_k_data['industry_id_level1_macd_positive'] = tmp['macd'].apply(lambda x: 1 if x>0 else 0)
		industry_id_level1_k_data['industry_id_level1_macd_dif_ratio'] = (tmp['macd_dif']-tmp['macd_dif_min'])/(tmp['macd_dif_max']-tmp['macd_dif_min'])
		for day_cnt in [2, 3, 5, 10, 20, 40]:
			industry_id_level1_k_data['industry_id_level1_macd_dif_' + str(day_cnt)] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_id_level1_k_data['industry_id_level1_macd_dea_' + str(day_cnt)] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_id_level1_k_data['industry_id_level1_macd_' + str(day_cnt)] = tmp['macd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_id_level1_k_data['industry_id_level1_macd_positive_ratio_' + str(day_cnt)] = industry_id_level1_k_data['industry_id_level1_macd_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
		industry_id_level1_k_data['industry_id_level1_macd_dif_dea'] = (tmp['macd_dif']-tmp['macd_dea']).groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1)))

		# boll线
		tmp['mb_20'] = tmp["industry_id_level1_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).mean())
		tmp['md_20'] = tmp["industry_id_level1_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).std())
		tmp['up_20'] = tmp['mb_20'] + 2 * tmp['md_20']
		tmp['dn_20'] = tmp['mb_20'] - 2 * tmp['md_20']
		industry_id_level1_k_data['industry_id_level1_width_20'] = 4 * tmp['md_20'] / tmp['mb_20']
		industry_id_level1_k_data['industry_id_level1_close_mb20_diff'] = (tmp["industry_id_level1_close"] - tmp['mb_20'])/(2 * tmp['md_20'])

		# cr指标
		tmp = industry_id_level1_k_data[["industry_id_level1_close", "industry_id_level1_open", "industry_id_level1_high", "industry_id_level1_low"]]
		tmp['cr_m'] = (tmp["industry_id_level1_close"] + tmp["industry_id_level1_open"] + tmp["industry_id_level1_high"] + tmp["industry_id_level1_low"])/4
		tmp['cr_ym'] = tmp['cr_m'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0]))
		tmp['cr_p1_day'] = (tmp["industry_id_level1_high"] - tmp['cr_ym'])
		tmp['cr_p2_day'] = (tmp['cr_ym'] - tmp["industry_id_level1_low"])
		for day_cnt in (3, 5, 10, 20, 40):
			tmp['cr_p1_' + str(day_cnt) + 'd'] = tmp['cr_p1_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			tmp['cr_p2_' + str(day_cnt) + 'd'] = tmp['cr_p2_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			industry_id_level1_k_data['industry_id_level1_cr_' + str(day_cnt) + 'd'] = tmp['cr_p1_' + str(day_cnt) + 'd'] / tmp['cr_p2_' + str(day_cnt) + 'd']
			industry_id_level1_k_data['industry_id_level1_cr_' + str(day_cnt) + 'd'] = industry_id_level1_k_data['industry_id_level1_cr_' + str(day_cnt) + 'd'].map(lambda x: float2Bucket(float(x) * 100, 0.2, 0, 500, 100))
		# rsi指标
		tmp = industry_id_level1_k_data[["industry_id_level1_close", "industry_id_level1_preclose"]]
		tmp['price_dif'] = tmp["industry_id_level1_close"] - tmp["industry_id_level1_preclose"]
		tmp['rsi_positive'] = tmp['price_dif'].apply(lambda x: max(x, 0))
		tmp['rsi_all'] = tmp['price_dif'].apply(lambda x: abs(x))
		for day_cnt in (3, 5, 10, 20, 40):
			tmp['rsi_positive_sum'] = tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			tmp['rsi_all_sum'] = tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			industry_id_level1_k_data['industry_id_level1_rsi_' + str(day_cnt) + 'd'] =  tmp['rsi_positive_sum'] / tmp['rsi_all_sum']

		tmp = industry_id_level1_k_data[["industry_id_level1_turn", "industry_id_level1_close"]]
		day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
		for index in range(len(day_cnt_list)):
			day_cnt = day_cnt_list[index]
			day_cnt_last = day_cnt_list[index-1]
			industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level1_k_data["industry_id_level1_turn"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level1_k_data["industry_id_level1_turn"]/industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg']
			industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			if index>0:
				industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + 'd' + '_avg']/industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg']

			tmp['industry_id_level1_close_' + str(day_cnt) + 'd' + '_avg'] = tmp["industry_id_level1_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level1_k_data["industry_id_level1_close"]/tmp['industry_id_level1_close_' + str(day_cnt) + 'd' + '_avg']
			industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level1_k_data["industry_id_level1_close"]/industry_id_level1_k_data["industry_id_level1_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level1_k_data["industry_id_level1_close"]/industry_id_level1_k_data["industry_id_level1_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + '_dif'] = industry_id_level1_k_data["industry_id_level1_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 1.0*(y[day_cnt-1])/y[0]))
			if index>0:
				industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level1_close_' + str(day_cnt_last) + 'd' + '_avg']/tmp['industry_id_level1_close_' + str(day_cnt) + 'd' + '_avg']

		for index in range(len(day_cnt_list)):
			day_cnt = day_cnt_list[index]
			day_cnt_last = day_cnt_list[index-1]
			if index > 0:
				industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 20, 0, 10, 200))
				industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 4, 200))
			industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 2, 0, 100, 200))
			industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
			industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 10, 1, 50, 500))
			industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

			industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
			industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 100, 0, 1, 100))
			industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 1, 10, 500))
			industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + '_dif'] = industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + '_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

		industry_id_level1_k_data = industry_id_level1_k_data.reset_index(level=0, drop=False)
		industry_id_level1_k_data = industry_id_level1_k_data.reset_index(level=0, drop=False)

		feature_all = pd.merge(feature_all, industry_id_level1_k_data, how="left", left_on=["date",'industry_id_level1'],right_on=["date",'industry_id_level1'])
		feature_all = feature_all.sort_values(['date', 'code'])
		del industry_id_level1_k_data
		gc.collect()

		# 二级行业特征
		raw_k_data = pd.read_csv(self.k_file_path)
		raw_k_data_his = pd.read_csv(self.k_file_path_his)
		raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
		del raw_k_data_his
		gc.collect()
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
		raw_k_data_tmp = raw_k_data[["date","industry_id_level2", "volume_total"]].groupby(["date","industry_id_level2"]).sum()
		raw_k_data_tmp.columns = ['industry_id_level2_volume_total']
		raw_k_data_tmp['industry_id_level2_rise_ratio'] = raw_k_data[["date","industry_id_level2", "rise"]].groupby(["date","industry_id_level2"]).mean()
		raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
		raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
		raw_k_data = pd.merge(raw_k_data, raw_k_data_tmp, how="left", left_on=["date","industry_id_level2"], right_on=["date","industry_id_level2"])
		raw_k_data['volume_total_ratio'] = raw_k_data["volume_total"]/raw_k_data["industry_id_level2_volume_total"]

		raw_k_data["open"] = raw_k_data["open"]*raw_k_data['volume_total_ratio']
		raw_k_data["close"] = raw_k_data["close"]*raw_k_data['volume_total_ratio']
		raw_k_data["preclose"] = raw_k_data["preclose"]*raw_k_data['volume_total_ratio']
		raw_k_data["high"] = raw_k_data["high"]*raw_k_data['volume_total_ratio']
		raw_k_data["low"] = raw_k_data["low"]*raw_k_data['volume_total_ratio']
		raw_k_data["turn"] = raw_k_data["turn"]*raw_k_data['volume_total_ratio']
		raw_k_data["amount"] = raw_k_data["amount"]*raw_k_data['volume_total_ratio']
		raw_k_data['pctChg'] = raw_k_data['pctChg'].map(lambda x: x/100.0)
		raw_k_data["pctChg"] = raw_k_data["pctChg"]*raw_k_data['volume_total_ratio']
		raw_k_data["peTTM"] = raw_k_data["peTTM"]*raw_k_data['volume_total_ratio']
		raw_k_data["pcfNcfTTM"] = raw_k_data["pcfNcfTTM"]*raw_k_data['volume_total_ratio']
		raw_k_data["pbMRQ"] = raw_k_data["pbMRQ"]*raw_k_data['volume_total_ratio']
		raw_k_data["industry_id_level2_rise_ratio"] = raw_k_data["industry_id_level2_rise_ratio"]*raw_k_data['volume_total_ratio']

		industry_id_level2_k_data = raw_k_data[["industry_id_level2","open","close","preclose","high","low","turn","date","amount","pctChg","peTTM","pcfNcfTTM","pbMRQ", 'industry_id_level2_rise_ratio']].groupby(['industry_id_level2','date']).sum()
		industry_id_level2_k_data.columns = ["industry_id_level2_open","industry_id_level2_close","industry_id_level2_preclose","industry_id_level2_high","industry_id_level2_low","industry_id_level2_turn","industry_id_level2_amount","industry_id_level2_pctChg","industry_id_level2_peTTM","industry_id_level2_pcfNcfTTM","industry_id_level2_pbMRQ", 'industry_id_level2_rise_ratio']

		industry_id_level2_k_data['industry_id_level2_open_ratio'] = ((industry_id_level2_k_data["industry_id_level2_open"] - industry_id_level2_k_data["industry_id_level2_preclose"]) / industry_id_level2_k_data["industry_id_level2_preclose"])
		industry_id_level2_k_data['industry_id_level2_close_ratio'] = ((industry_id_level2_k_data["industry_id_level2_close"] - industry_id_level2_k_data["industry_id_level2_open"]) / industry_id_level2_k_data["industry_id_level2_open"])
		industry_id_level2_k_data['industry_id_level2_high_ratio'] = ((industry_id_level2_k_data["industry_id_level2_high"] - industry_id_level2_k_data["industry_id_level2_preclose"]) / industry_id_level2_k_data["industry_id_level2_preclose"])
		industry_id_level2_k_data['industry_id_level2_low_ratio'] = ((industry_id_level2_k_data["industry_id_level2_low"] - industry_id_level2_k_data["industry_id_level2_preclose"]) / industry_id_level2_k_data["industry_id_level2_preclose"])

		industry_id_level2_k_data['industry_id_level2_open_ratio_7d_avg'] = industry_id_level2_k_data['industry_id_level2_open_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_id_level2_k_data['industry_id_level2_close_ratio_7d_avg'] = industry_id_level2_k_data['industry_id_level2_close_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_id_level2_k_data['industry_id_level2_high_ratio_7d_avg'] = industry_id_level2_k_data['industry_id_level2_high_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_id_level2_k_data['industry_id_level2_low_ratio_7d_avg'] = industry_id_level2_k_data['industry_id_level2_low_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())

		industry_id_level2_k_data['industry_id_level2_turn'] = industry_id_level2_k_data["industry_id_level2_turn"].map(lambda x: float2Bucket(float(x), 1, 0, 200, 200))
		industry_id_level2_k_data['industry_id_level2_amount'] = industry_id_level2_k_data["industry_id_level2_amount"].map(lambda x: None if x == '' else float2Bucket(float(x) * 0.00000001, 1, 0, 10000, 10000))
		industry_id_level2_k_data['industry_id_level2_peTTM'] = industry_id_level2_k_data["industry_id_level2_peTTM"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 1500, 1500))
		industry_id_level2_k_data['industry_id_level2_pcfNcfTTM'] = industry_id_level2_k_data["industry_id_level2_pcfNcfTTM"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 2000, 2000))
		industry_id_level2_k_data['industry_id_level2_pbMRQ'] = industry_id_level2_k_data["industry_id_level2_pbMRQ"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 2000, 2000))


		# kdj 5, 9, 19, 36, 45, 73，
		# 任意初始化，超过30天后的kdj值基本一样
		tmp = industry_id_level2_k_data[["industry_id_level2_low", "industry_id_level2_high", "industry_id_level2_close"]]
		for day_cnt in [5, 9, 19, 73]:
		# industry_id_level2_k_data['industry_id_level2_rsv'] = industry_id_level2_k_data[["industry_id_level2_low", "industry_id_level2_high", "industry_id_level2_close"]].groupby(level=0).apply(lambda x: (x.close-x.low.rolling(min_periods=1, window=day_cnt, center=False).min())/(x.high.rolling(min_periods=1, window=day_cnt, center=False).max()-x.low.rolling(min_periods=1, window=day_cnt, center=False).min()))
			tmp['min'] = tmp["industry_id_level2_low"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			tmp['max'] = tmp["industry_id_level2_high"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_id_level2_k_data['industry_id_level2_rsv_' + str(day_cnt)] = (tmp["industry_id_level2_close"] - tmp['min'])/(tmp['max'] - tmp['min'])
			industry_id_level2_k_data['industry_id_level2_k_value_' + str(day_cnt)] = industry_id_level2_k_data['industry_id_level2_rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
			industry_id_level2_k_data['industry_id_level2_d_value_' + str(day_cnt)] = industry_id_level2_k_data['industry_id_level2_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
			industry_id_level2_k_data['industry_id_level2_j_value_' + str(day_cnt)] = 3 * industry_id_level2_k_data['industry_id_level2_k_value_' + str(day_cnt)] - 2 * industry_id_level2_k_data['industry_id_level2_d_value_' + str(day_cnt)]
			industry_id_level2_k_data['industry_id_level2_k_value_trend_' + str(day_cnt)] = industry_id_level2_k_data['industry_id_level2_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1])-y[0]))
			industry_id_level2_k_data['industry_id_level2_kd_value' + str(day_cnt)] = (industry_id_level2_k_data['industry_id_level2_k_value_' + str(day_cnt)] - industry_id_level2_k_data['industry_id_level2_d_value_' + str(day_cnt)]).rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1))

		# macd 12
		# 任意初始化，超过30天后的macd值基本一样
		tmp = industry_id_level2_k_data[['industry_id_level2_close']]
		tmp['ema_12'] = tmp['industry_id_level2_close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 13, adjust=False).mean())
		tmp['ema_26'] = tmp['industry_id_level2_close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 27, adjust=False).mean())
		tmp['macd_dif'] = tmp['ema_12'] - tmp['ema_26']
		tmp['macd_dea'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0 / 5, adjust=False).mean())
		tmp['macd_dif_max'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
		tmp['macd_dif_min'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
		tmp['macd_dea_max'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
		tmp['macd_dea_min'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
		tmp['macd'] = (tmp['macd_dif'] - tmp['macd_dea']) * 2
		industry_id_level2_k_data['industry_id_level2_macd_positive'] = tmp['macd'].apply(lambda x: 1 if x>0 else 0)
		industry_id_level2_k_data['industry_id_level2_macd_dif_ratio'] = (tmp['macd_dif']-tmp['macd_dif_min'])/(tmp['macd_dif_max']-tmp['macd_dif_min'])
		for day_cnt in [2, 3, 5, 10, 20, 40]:
			industry_id_level2_k_data['industry_id_level2_macd_dif_' + str(day_cnt)] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_id_level2_k_data['industry_id_level2_macd_dea_' + str(day_cnt)] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_id_level2_k_data['industry_id_level2_macd_' + str(day_cnt)] = tmp['macd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_id_level2_k_data['industry_id_level2_macd_positive_ratio_' + str(day_cnt)] = industry_id_level2_k_data['industry_id_level2_macd_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
		industry_id_level2_k_data['industry_id_level2_macd_dif_dea'] = (tmp['macd_dif']-tmp['macd_dea']).groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1)))

		# boll线
		tmp['mb_20'] = tmp["industry_id_level2_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).mean())
		tmp['md_20'] = tmp["industry_id_level2_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).std())
		tmp['up_20'] = tmp['mb_20'] + 2 * tmp['md_20']
		tmp['dn_20'] = tmp['mb_20'] - 2 * tmp['md_20']
		industry_id_level2_k_data['industry_id_level2_width_20'] = 4 * tmp['md_20'] / tmp['mb_20']
		industry_id_level2_k_data['industry_id_level2_close_mb20_diff'] = (tmp["industry_id_level2_close"] - tmp['mb_20'])/(2 * tmp['md_20'])

		# cr指标
		tmp = industry_id_level2_k_data[["industry_id_level2_close", "industry_id_level2_open", "industry_id_level2_high", "industry_id_level2_low"]]
		tmp['cr_m'] = (tmp["industry_id_level2_close"] + tmp["industry_id_level2_open"] + tmp["industry_id_level2_high"] + tmp["industry_id_level2_low"])/4
		tmp['cr_ym'] = tmp['cr_m'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0]))
		tmp['cr_p1_day'] = (tmp["industry_id_level2_high"] - tmp['cr_ym'])
		tmp['cr_p2_day'] = (tmp['cr_ym'] - tmp["industry_id_level2_low"])
		for day_cnt in (3, 5, 10, 20, 40):
			tmp['cr_p1_' + str(day_cnt) + 'd'] = tmp['cr_p1_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			tmp['cr_p2_' + str(day_cnt) + 'd'] = tmp['cr_p2_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			industry_id_level2_k_data['industry_id_level2_cr_' + str(day_cnt) + 'd'] = tmp['cr_p1_' + str(day_cnt) + 'd'] / tmp['cr_p2_' + str(day_cnt) + 'd']
			industry_id_level2_k_data['industry_id_level2_cr_' + str(day_cnt) + 'd'] = industry_id_level2_k_data['industry_id_level2_cr_' + str(day_cnt) + 'd'].map(lambda x: float2Bucket(float(x) * 100, 0.2, 0, 500, 100))
		# rsi指标
		tmp = industry_id_level2_k_data[["industry_id_level2_close", "industry_id_level2_preclose"]]
		tmp['price_dif'] = tmp["industry_id_level2_close"] - tmp["industry_id_level2_preclose"]
		tmp['rsi_positive'] = tmp['price_dif'].apply(lambda x: max(x, 0))
		tmp['rsi_all'] = tmp['price_dif'].apply(lambda x: abs(x))
		for day_cnt in (3, 5, 10, 20, 40):
			tmp['rsi_positive_sum'] = tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			tmp['rsi_all_sum'] = tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			industry_id_level2_k_data['industry_id_level2_rsi_' + str(day_cnt) + 'd'] =  tmp['rsi_positive_sum'] / tmp['rsi_all_sum']

		tmp = industry_id_level2_k_data[["industry_id_level2_turn", "industry_id_level2_close"]]
		day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
		for index in range(len(day_cnt_list)):
			day_cnt = day_cnt_list[index]
			day_cnt_last = day_cnt_list[index-1]
			industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level2_k_data["industry_id_level2_turn"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level2_k_data["industry_id_level2_turn"]/industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg']
			industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			if index>0:
				industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + 'd' + '_avg']/industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg']

			tmp['industry_id_level2_close_' + str(day_cnt) + 'd' + '_avg'] = tmp["industry_id_level2_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level2_k_data["industry_id_level2_close"]/tmp['industry_id_level2_close_' + str(day_cnt) + 'd' + '_avg']
			industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level2_k_data["industry_id_level2_close"]/industry_id_level2_k_data["industry_id_level2_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level2_k_data["industry_id_level2_close"]/industry_id_level2_k_data["industry_id_level2_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + '_dif'] = industry_id_level2_k_data["industry_id_level2_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 1.0*(y[day_cnt-1])/y[0]))
			if index>0:
				industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level2_close_' + str(day_cnt_last) + 'd' + '_avg']/tmp['industry_id_level2_close_' + str(day_cnt) + 'd' + '_avg']

		for index in range(len(day_cnt_list)):
			day_cnt = day_cnt_list[index]
			day_cnt_last = day_cnt_list[index-1]
			if index > 0:
				industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 20, 0, 10, 200))
				industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 4, 200))
			industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 2, 0, 100, 200))
			industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
			industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 10, 1, 50, 500))
			industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

			industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
			industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 100, 0, 1, 100))
			industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 1, 10, 500))
			industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + '_dif'] = industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + '_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

		industry_id_level2_k_data = industry_id_level2_k_data.reset_index(level=0, drop=False)
		industry_id_level2_k_data = industry_id_level2_k_data.reset_index(level=0, drop=False)

		feature_all = pd.merge(feature_all, industry_id_level2_k_data, how="left", left_on=["date",'industry_id_level2'],right_on=["date",'industry_id_level2'])
		feature_all = feature_all.sort_values(['date', 'code'])
		del industry_id_level2_k_data
		gc.collect()

		# 三级行业特征
		raw_k_data = pd.read_csv(self.k_file_path)
		raw_k_data_his = pd.read_csv(self.k_file_path_his)
		raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
		del raw_k_data_his
		gc.collect()
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
		raw_k_data_tmp = raw_k_data[["date","industry_id_level3", "volume_total"]].groupby(["date","industry_id_level3"]).sum()
		raw_k_data_tmp.columns = ['industry_id_level3_volume_total']
		raw_k_data_tmp['industry_id_level3_rise_ratio'] = raw_k_data[["date","industry_id_level3", "rise"]].groupby(["date","industry_id_level3"]).mean()
		raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
		raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
		raw_k_data = pd.merge(raw_k_data, raw_k_data_tmp, how="left", left_on=["date","industry_id_level3"], right_on=["date","industry_id_level3"])
		raw_k_data['volume_total_ratio'] = raw_k_data["volume_total"]/raw_k_data["industry_id_level3_volume_total"]

		raw_k_data["open"] = raw_k_data["open"]*raw_k_data['volume_total_ratio']
		raw_k_data["close"] = raw_k_data["close"]*raw_k_data['volume_total_ratio']
		raw_k_data["preclose"] = raw_k_data["preclose"]*raw_k_data['volume_total_ratio']
		raw_k_data["high"] = raw_k_data["high"]*raw_k_data['volume_total_ratio']
		raw_k_data["low"] = raw_k_data["low"]*raw_k_data['volume_total_ratio']
		raw_k_data["turn"] = raw_k_data["turn"]*raw_k_data['volume_total_ratio']
		raw_k_data["amount"] = raw_k_data["amount"]*raw_k_data['volume_total_ratio']
		raw_k_data['pctChg'] = raw_k_data['pctChg'].map(lambda x: x/100.0)
		raw_k_data["pctChg"] = raw_k_data["pctChg"]*raw_k_data['volume_total_ratio']
		raw_k_data["peTTM"] = raw_k_data["peTTM"]*raw_k_data['volume_total_ratio']
		raw_k_data["pcfNcfTTM"] = raw_k_data["pcfNcfTTM"]*raw_k_data['volume_total_ratio']
		raw_k_data["pbMRQ"] = raw_k_data["pbMRQ"]*raw_k_data['volume_total_ratio']
		raw_k_data["industry_id_level3_rise_ratio"] = raw_k_data["industry_id_level3_rise_ratio"]*raw_k_data['volume_total_ratio']

		industry_id_level3_k_data = raw_k_data[["industry_id_level3","open","close","preclose","high","low","turn","date","amount","pctChg","peTTM","pcfNcfTTM","pbMRQ", 'industry_id_level3_rise_ratio']].groupby(['industry_id_level3','date']).sum()
		industry_id_level3_k_data.columns = ["industry_id_level3_open","industry_id_level3_close","industry_id_level3_preclose","industry_id_level3_high","industry_id_level3_low","industry_id_level3_turn","industry_id_level3_amount","industry_id_level3_pctChg","industry_id_level3_peTTM","industry_id_level3_pcfNcfTTM","industry_id_level3_pbMRQ", 'industry_id_level3_rise_ratio']

		industry_id_level3_k_data['industry_id_level3_open_ratio'] = ((industry_id_level3_k_data["industry_id_level3_open"] - industry_id_level3_k_data["industry_id_level3_preclose"]) / industry_id_level3_k_data["industry_id_level3_preclose"])
		industry_id_level3_k_data['industry_id_level3_close_ratio'] = ((industry_id_level3_k_data["industry_id_level3_close"] - industry_id_level3_k_data["industry_id_level3_open"]) / industry_id_level3_k_data["industry_id_level3_open"])
		industry_id_level3_k_data['industry_id_level3_high_ratio'] = ((industry_id_level3_k_data["industry_id_level3_high"] - industry_id_level3_k_data["industry_id_level3_preclose"]) / industry_id_level3_k_data["industry_id_level3_preclose"])
		industry_id_level3_k_data['industry_id_level3_low_ratio'] = ((industry_id_level3_k_data["industry_id_level3_low"] - industry_id_level3_k_data["industry_id_level3_preclose"]) / industry_id_level3_k_data["industry_id_level3_preclose"])

		industry_id_level3_k_data['industry_id_level3_open_ratio_7d_avg'] = industry_id_level3_k_data['industry_id_level3_open_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_id_level3_k_data['industry_id_level3_close_ratio_7d_avg'] = industry_id_level3_k_data['industry_id_level3_close_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_id_level3_k_data['industry_id_level3_high_ratio_7d_avg'] = industry_id_level3_k_data['industry_id_level3_high_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_id_level3_k_data['industry_id_level3_low_ratio_7d_avg'] = industry_id_level3_k_data['industry_id_level3_low_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())

		industry_id_level3_k_data['industry_id_level3_turn'] = industry_id_level3_k_data["industry_id_level3_turn"].map(lambda x: float2Bucket(float(x), 1, 0, 200, 200))
		industry_id_level3_k_data['industry_id_level3_amount'] = industry_id_level3_k_data["industry_id_level3_amount"].map(lambda x: None if x == '' else float2Bucket(float(x) * 0.00000001, 1, 0, 10000, 10000))
		industry_id_level3_k_data['industry_id_level3_peTTM'] = industry_id_level3_k_data["industry_id_level3_peTTM"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 1500, 1500))
		industry_id_level3_k_data['industry_id_level3_pcfNcfTTM'] = industry_id_level3_k_data["industry_id_level3_pcfNcfTTM"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 2000, 2000))
		industry_id_level3_k_data['industry_id_level3_pbMRQ'] = industry_id_level3_k_data["industry_id_level3_pbMRQ"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 2000, 2000))


		# kdj 5, 9, 19, 36, 45, 73，
		# 任意初始化，超过30天后的kdj值基本一样
		tmp = industry_id_level3_k_data[["industry_id_level3_low", "industry_id_level3_high", "industry_id_level3_close"]]
		for day_cnt in [5, 9, 19, 73]:
		# industry_id_level3_k_data['industry_id_level3_rsv'] = industry_id_level3_k_data[["industry_id_level3_low", "industry_id_level3_high", "industry_id_level3_close"]].groupby(level=0).apply(lambda x: (x.close-x.low.rolling(min_periods=1, window=day_cnt, center=False).min())/(x.high.rolling(min_periods=1, window=day_cnt, center=False).max()-x.low.rolling(min_periods=1, window=day_cnt, center=False).min()))
			tmp['min'] = tmp["industry_id_level3_low"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			tmp['max'] = tmp["industry_id_level3_high"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_id_level3_k_data['industry_id_level3_rsv_' + str(day_cnt)] = (tmp["industry_id_level3_close"] - tmp['min'])/(tmp['max'] - tmp['min'])
			industry_id_level3_k_data['industry_id_level3_k_value_' + str(day_cnt)] = industry_id_level3_k_data['industry_id_level3_rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
			industry_id_level3_k_data['industry_id_level3_d_value_' + str(day_cnt)] = industry_id_level3_k_data['industry_id_level3_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
			industry_id_level3_k_data['industry_id_level3_j_value_' + str(day_cnt)] = 3 * industry_id_level3_k_data['industry_id_level3_k_value_' + str(day_cnt)] - 2 * industry_id_level3_k_data['industry_id_level3_d_value_' + str(day_cnt)]
			industry_id_level3_k_data['industry_id_level3_k_value_trend_' + str(day_cnt)] = industry_id_level3_k_data['industry_id_level3_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1])-y[0]))
			industry_id_level3_k_data['industry_id_level3_kd_value' + str(day_cnt)] = (industry_id_level3_k_data['industry_id_level3_k_value_' + str(day_cnt)] - industry_id_level3_k_data['industry_id_level3_d_value_' + str(day_cnt)]).rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1))

		# macd 12
		# 任意初始化，超过30天后的macd值基本一样
		tmp = industry_id_level3_k_data[['industry_id_level3_close']]
		tmp['ema_12'] = tmp['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 13, adjust=False).mean())
		tmp['ema_26'] = tmp['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 27, adjust=False).mean())
		tmp['macd_dif'] = tmp['ema_12'] - tmp['ema_26']
		tmp['macd_dea'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0 / 5, adjust=False).mean())
		tmp['macd_dif_max'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
		tmp['macd_dif_min'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
		tmp['macd_dea_max'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
		tmp['macd_dea_min'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
		tmp['macd'] = (tmp['macd_dif'] - tmp['macd_dea']) * 2
		industry_id_level3_k_data['industry_id_level3_macd_positive'] = tmp['macd'].apply(lambda x: 1 if x>0 else 0)
		industry_id_level3_k_data['industry_id_level3_macd_dif_ratio'] = (tmp['macd_dif']-tmp['macd_dif_min'])/(tmp['macd_dif_max']-tmp['macd_dif_min'])
		for day_cnt in [2, 3, 5, 10, 20, 40]:
			industry_id_level3_k_data['industry_id_level3_macd_dif_' + str(day_cnt)] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_id_level3_k_data['industry_id_level3_macd_dea_' + str(day_cnt)] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_id_level3_k_data['industry_id_level3_macd_' + str(day_cnt)] = tmp['macd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_id_level3_k_data['industry_id_level3_macd_positive_ratio_' + str(day_cnt)] = industry_id_level3_k_data['industry_id_level3_macd_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
		industry_id_level3_k_data['industry_id_level3_macd_dif_dea'] = (tmp['macd_dif']-tmp['macd_dea']).groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1)))

		# boll线
		tmp['mb_20'] = tmp["industry_id_level3_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).mean())
		tmp['md_20'] = tmp["industry_id_level3_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).std())
		tmp['up_20'] = tmp['mb_20'] + 2 * tmp['md_20']
		tmp['dn_20'] = tmp['mb_20'] - 2 * tmp['md_20']
		industry_id_level3_k_data['industry_id_level3_width_20'] = 4 * tmp['md_20'] / tmp['mb_20']
		industry_id_level3_k_data['industry_id_level3_close_mb20_diff'] = (tmp["industry_id_level3_close"] - tmp['mb_20'])/(2 * tmp['md_20'])

		# cr指标
		tmp = industry_id_level3_k_data[["industry_id_level3_close", "industry_id_level3_open", "industry_id_level3_high", "industry_id_level3_low"]]
		tmp['cr_m'] = (tmp["industry_id_level3_close"] + tmp["industry_id_level3_open"] + tmp["industry_id_level3_high"] + tmp["industry_id_level3_low"])/4
		tmp['cr_ym'] = tmp['cr_m'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0]))
		tmp['cr_p1_day'] = (tmp["industry_id_level3_high"] - tmp['cr_ym'])
		tmp['cr_p2_day'] = (tmp['cr_ym'] - tmp["industry_id_level3_low"])
		for day_cnt in (3, 5, 10, 20, 40):
			tmp['cr_p1_' + str(day_cnt) + 'd'] = tmp['cr_p1_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			tmp['cr_p2_' + str(day_cnt) + 'd'] = tmp['cr_p2_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			industry_id_level3_k_data['industry_id_level3_cr_' + str(day_cnt) + 'd'] = tmp['cr_p1_' + str(day_cnt) + 'd'] / tmp['cr_p2_' + str(day_cnt) + 'd']
			industry_id_level3_k_data['industry_id_level3_cr_' + str(day_cnt) + 'd'] = industry_id_level3_k_data['industry_id_level3_cr_' + str(day_cnt) + 'd'].map(lambda x: float2Bucket(float(x) * 100, 0.2, 0, 500, 100))
		# rsi指标
		tmp = industry_id_level3_k_data[["industry_id_level3_close", "industry_id_level3_preclose"]]
		tmp['price_dif'] = tmp["industry_id_level3_close"] - tmp["industry_id_level3_preclose"]
		tmp['rsi_positive'] = tmp['price_dif'].apply(lambda x: max(x, 0))
		tmp['rsi_all'] = tmp['price_dif'].apply(lambda x: abs(x))
		for day_cnt in (3, 5, 10, 20, 40):
			tmp['rsi_positive_sum'] = tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			tmp['rsi_all_sum'] = tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			industry_id_level3_k_data['industry_id_level3_rsi_' + str(day_cnt) + 'd'] =  tmp['rsi_positive_sum'] / tmp['rsi_all_sum']

		tmp = industry_id_level3_k_data[["industry_id_level3_turn", "industry_id_level3_close"]]
		day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
		for index in range(len(day_cnt_list)):
			day_cnt = day_cnt_list[index]
			day_cnt_last = day_cnt_list[index-1]
			industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level3_k_data["industry_id_level3_turn"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level3_k_data["industry_id_level3_turn"]/industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg']
			industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			if index>0:
				industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + 'd' + '_avg']/industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg']

			tmp['industry_id_level3_close_' + str(day_cnt) + 'd' + '_avg'] = tmp["industry_id_level3_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level3_k_data["industry_id_level3_close"]/tmp['industry_id_level3_close_' + str(day_cnt) + 'd' + '_avg']
			industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level3_k_data["industry_id_level3_close"]/industry_id_level3_k_data["industry_id_level3_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level3_k_data["industry_id_level3_close"]/industry_id_level3_k_data["industry_id_level3_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + '_dif'] = industry_id_level3_k_data["industry_id_level3_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 1.0*(y[day_cnt-1])/y[0]))
			if index>0:
				industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level3_close_' + str(day_cnt_last) + 'd' + '_avg']/tmp['industry_id_level3_close_' + str(day_cnt) + 'd' + '_avg']

		for index in range(len(day_cnt_list)):
			day_cnt = day_cnt_list[index]
			day_cnt_last = day_cnt_list[index-1]
			if index > 0:
				industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 20, 0, 10, 200))
				industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 4, 200))
			industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 2, 0, 100, 200))
			industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
			industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 10, 1, 50, 500))
			industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

			industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
			industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 100, 0, 1, 100))
			industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 1, 10, 500))
			industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + '_dif'] = industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + '_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

		industry_id_level3_k_data = industry_id_level3_k_data.reset_index(level=0, drop=False)
		industry_id_level3_k_data = industry_id_level3_k_data.reset_index(level=0, drop=False)

		feature_all = pd.merge(feature_all, industry_id_level3_k_data, how="left", left_on=["date",'industry_id_level3'],right_on=["date",'industry_id_level3'])
		feature_all = feature_all.sort_values(['date', 'code'])
		del industry_id_level3_k_data
		gc.collect()

		# 大盘趋势类特征
		raw_k_data = pd.read_csv(self.k_file_path)
		raw_k_data_his = pd.read_csv(self.k_file_path_his)
		raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
		raw_k_data = raw_k_data[raw_k_data['industry_id_level3'] > 0]
		del raw_k_data_his
		gc.collect()
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
		raw_k_data['pctChg'] = raw_k_data['pctChg'].map(lambda x: x / 100.0)
		raw_k_data['rise_ratio'] = raw_k_data['pctChg'].map(lambda x: 1 if x>0 else 0)
		raw_k_data['peTTM'] = raw_k_data['peTTM']
		raw_k_data['pcfNcfTTM'] = raw_k_data['pcfNcfTTM']
		raw_k_data['pbMRQ'] = raw_k_data['pbMRQ']
		raw_k_data['isST'] = raw_k_data['isST']

		all_data_raw = raw_k_data[['date', 'pctChg', 'turn', 'rise_ratio']].groupby('date').mean()
		all_data_raw.columns = ['all_pctChg', 'all_turn', 'all_rise_ratio']
		# all_data_raw = all_data_raw.reset_index(level=0, drop=False)
		day_cnt_list = [3, 5, 10]
		for index in range(len(day_cnt_list)):
			day_cnt = day_cnt_list[index]
			all_data_raw['all_pctChg_' + str(day_cnt)] = all_data_raw['all_pctChg'].rolling(min_periods=1, window=day_cnt, center=False).mean()
			all_data_raw['all_turn_' + str(day_cnt)] = all_data_raw['all_turn'].rolling(min_periods=1, window=day_cnt, center=False).mean()
			all_data_raw['all_rise_ratio_' + str(day_cnt)] = all_data_raw['all_rise_ratio'].rolling(min_periods=1, window=day_cnt, center=False).mean()
		all_data_raw['all_turn'] = all_data_raw['all_turn'].map(lambda x: float2Bucket(float(x), 1, 0, 200, 200))
		for index in range(len(day_cnt_list)):
			day_cnt = day_cnt_list[index]
			all_data_raw['all_turn_' + str(day_cnt)] = all_data_raw['all_turn_' + str(day_cnt)].map(lambda x: float2Bucket(float(x), 1, 0, 200, 200))

		all_data_raw = all_data_raw.reset_index(level=0, drop=False)
		zs_data = feature_all[(feature_all['code'] == 'sh.000001') | (feature_all['code'] == 'sz.399001') | (feature_all['code'] == 'sz.399006')]
		zs_data = zs_data[['date','code','open_ratio','close_ratio','high_ratio','low_ratio','pctChg','open_ratio_7d_avg','close_ratio_7d_avg','high_ratio_7d_avg','low_ratio_7d_avg','amount','peTTM','pcfNcfTTM','pbMRQ','isST','turn','rsv_5','k_value_5','d_value_5','j_value_5','k_value_trend_5','kd_value5','rsv_9','k_value_9','d_value_9','j_value_9','k_value_trend_9','kd_value9','rsv_19','k_value_19','d_value_19','j_value_19','k_value_trend_19','kd_value19','rsv_73','k_value_73','d_value_73','j_value_73','k_value_trend_73','kd_value73','macd_positive','macd_dif_ratio','macd_dif_2','macd_dea_2','macd_2','macd_positive_ratio_2','macd_dif_3','macd_dea_3','macd_3','macd_positive_ratio_3','macd_dif_5','macd_dea_5','macd_5','macd_positive_ratio_5','macd_dif_10','macd_dea_10','macd_10','macd_positive_ratio_10','macd_dif_20','macd_dea_20','macd_20','macd_positive_ratio_20','macd_dif_40','macd_dea_40','macd_40','macd_positive_ratio_40','macd_dif_dea','width_20','close_mb20_diff','cr_3d','cr_5d','cr_10d','cr_20d','cr_40d','rsi_3d','rsi_5d','rsi_10d','rsi_20d','rsi_40d','turn_3d_avg','turn_3davg_dif','turn_3dmax_dif','turn_3dmin_dif','close_3davg_dif','close_3dmax_dif','close_3dmin_dif','close_3d_dif','turn_5d_avg','turn_5davg_dif','turn_5dmax_dif','turn_5dmin_dif','turn_3_5d_avg','close_5davg_dif','close_5dmax_dif','close_5dmin_dif','close_5d_dif','close_3_5d_avg','turn_10d_avg','turn_10davg_dif','turn_10dmax_dif','turn_10dmin_dif','turn_5_10d_avg','close_10davg_dif','close_10dmax_dif','close_10dmin_dif','close_10d_dif','close_5_10d_avg','turn_20d_avg','turn_20davg_dif','turn_20dmax_dif','turn_20dmin_dif','turn_10_20d_avg','close_20davg_dif','close_20dmax_dif','close_20dmin_dif','close_20d_dif','close_10_20d_avg','turn_30d_avg','turn_30davg_dif','turn_30dmax_dif','turn_30dmin_dif','turn_20_30d_avg','close_30davg_dif','close_30dmax_dif','close_30dmin_dif','close_30d_dif','close_20_30d_avg','turn_60d_avg','turn_60davg_dif','turn_60dmax_dif','turn_60dmin_dif','turn_30_60d_avg','close_60davg_dif','close_60dmax_dif','close_60dmin_dif','close_60d_dif','close_30_60d_avg','turn_120d_avg','turn_120davg_dif','turn_120dmax_dif','turn_120dmin_dif','turn_60_120d_avg','close_120davg_dif','close_120dmax_dif','close_120dmin_dif','close_120d_dif','close_60_120d_avg','turn_240d_avg','turn_240davg_dif','turn_240dmax_dif','turn_240dmin_dif','turn_120_240d_avg','close_240davg_dif','close_240dmax_dif','close_240dmin_dif','close_240d_dif','close_120_240d_avg']]
		zs_data['code_market'] = zs_data['code'].map(lambda x: self.tools.zh_code_market(x))
		zs_data = zs_data[['date','code_market','open_ratio','close_ratio','high_ratio','low_ratio','pctChg','open_ratio_7d_avg','close_ratio_7d_avg','high_ratio_7d_avg','low_ratio_7d_avg','amount','turn','rsv_5','k_value_5','d_value_5','j_value_5','k_value_trend_5','kd_value5','rsv_9','k_value_9','d_value_9','j_value_9','k_value_trend_9','kd_value9','rsv_19','k_value_19','d_value_19','j_value_19','k_value_trend_19','kd_value19','rsv_73','k_value_73','d_value_73','j_value_73','k_value_trend_73','kd_value73','macd_positive','macd_dif_ratio','macd_dif_2','macd_dea_2','macd_2','macd_positive_ratio_2','macd_dif_3','macd_dea_3','macd_3','macd_positive_ratio_3','macd_dif_5','macd_dea_5','macd_5','macd_positive_ratio_5','macd_dif_10','macd_dea_10','macd_10','macd_positive_ratio_10','macd_dif_20','macd_dea_20','macd_20','macd_positive_ratio_20','macd_dif_40','macd_dea_40','macd_40','macd_positive_ratio_40','macd_dif_dea','width_20','close_mb20_diff','cr_3d','cr_5d','cr_10d','cr_20d','cr_40d','rsi_3d','rsi_5d','rsi_10d','rsi_20d','rsi_40d','turn_3d_avg','turn_3davg_dif','turn_3dmax_dif','turn_3dmin_dif','close_3davg_dif','close_3dmax_dif','close_3dmin_dif','close_3d_dif','turn_5d_avg','turn_5davg_dif','turn_5dmax_dif','turn_5dmin_dif','turn_3_5d_avg','close_5davg_dif','close_5dmax_dif','close_5dmin_dif','close_5d_dif','close_3_5d_avg','turn_10d_avg','turn_10davg_dif','turn_10dmax_dif','turn_10dmin_dif','turn_5_10d_avg','close_10davg_dif','close_10dmax_dif','close_10dmin_dif','close_10d_dif','close_5_10d_avg','turn_20d_avg','turn_20davg_dif','turn_20dmax_dif','turn_20dmin_dif','turn_10_20d_avg','close_20davg_dif','close_20dmax_dif','close_20dmin_dif','close_20d_dif','close_10_20d_avg','turn_30d_avg','turn_30davg_dif','turn_30dmax_dif','turn_30dmin_dif','turn_20_30d_avg','close_30davg_dif','close_30dmax_dif','close_30dmin_dif','close_30d_dif','close_20_30d_avg','turn_60d_avg','turn_60davg_dif','turn_60dmax_dif','turn_60dmin_dif','turn_30_60d_avg','close_60davg_dif','close_60dmax_dif','close_60dmin_dif','close_60d_dif','close_30_60d_avg','turn_120d_avg','turn_120davg_dif','turn_120dmax_dif','turn_120dmin_dif','turn_60_120d_avg','close_120davg_dif','close_120dmax_dif','close_120dmin_dif','close_120d_dif','close_60_120d_avg','turn_240d_avg','turn_240davg_dif','turn_240dmax_dif','turn_240dmin_dif','turn_120_240d_avg','close_240davg_dif','close_240dmax_dif','close_240dmin_dif','close_240d_dif','close_120_240d_avg']]
		zs_data.columns = ['date','code_market','zs_open_ratio','zs_close_ratio','zs_high_ratio','zs_low_ratio','zs_pctChg','zs_open_ratio_7d_avg','zs_close_ratio_7d_avg','zs_high_ratio_7d_avg','zs_low_ratio_7d_avg','zs_amount','zs_turn','zs_rsv_5','zs_k_value_5','zs_d_value_5','zs_j_value_5','zs_k_value_trend_5','zs_kd_value5','zs_rsv_9','zs_k_value_9','zs_d_value_9','zs_j_value_9','zs_k_value_trend_9','zs_kd_value9','zs_rsv_19','zs_k_value_19','zs_d_value_19','zs_j_value_19','zs_k_value_trend_19','zs_kd_value19','zs_rsv_73','zs_k_value_73','zs_d_value_73','zs_j_value_73','zs_k_value_trend_73','zs_kd_value73','zs_macd_positive','zs_macd_dif_ratio','zs_macd_dif_2','zs_macd_dea_2','zs_macd_2','zs_macd_positive_ratio_2','zs_macd_dif_3','zs_macd_dea_3','zs_macd_3','zs_macd_positive_ratio_3','zs_macd_dif_5','zs_macd_dea_5','zs_macd_5','zs_macd_positive_ratio_5','zs_macd_dif_10','zs_macd_dea_10','zs_macd_10','zs_macd_positive_ratio_10','zs_macd_dif_20','zs_macd_dea_20','zs_macd_20','zs_macd_positive_ratio_20','zs_macd_dif_40','zs_macd_dea_40','zs_macd_40','zs_macd_positive_ratio_40','zs_macd_dif_dea','zs_width_20','zs_close_mb20_diff','zs_cr_3d','zs_cr_5d','zs_cr_10d','zs_cr_20d','zs_cr_40d','zs_rsi_3d','zs_rsi_5d','zs_rsi_10d','zs_rsi_20d','zs_rsi_40d','zs_turn_3d_avg','zs_turn_3davg_dif','zs_turn_3dmax_dif','zs_turn_3dmin_dif','zs_close_3davg_dif','zs_close_3dmax_dif','zs_close_3dmin_dif','zs_close_3d_dif','zs_turn_5d_avg','zs_turn_5davg_dif','zs_turn_5dmax_dif','zs_turn_5dmin_dif','zs_turn_3_5d_avg','zs_close_5davg_dif','zs_close_5dmax_dif','zs_close_5dmin_dif','zs_close_5d_dif','zs_close_3_5d_avg','zs_turn_10d_avg','zs_turn_10davg_dif','zs_turn_10dmax_dif','zs_turn_10dmin_dif','zs_turn_5_10d_avg','zs_close_10davg_dif','zs_close_10dmax_dif','zs_close_10dmin_dif','zs_close_10d_dif','zs_close_5_10d_avg','zs_turn_20d_avg','zs_turn_20davg_dif','zs_turn_20dmax_dif','zs_turn_20dmin_dif','zs_turn_10_20d_avg','zs_close_20davg_dif','zs_close_20dmax_dif','zs_close_20dmin_dif','zs_close_20d_dif','zs_close_10_20d_avg','zs_turn_30d_avg','zs_turn_30davg_dif','zs_turn_30dmax_dif','zs_turn_30dmin_dif','zs_turn_20_30d_avg','zs_close_30davg_dif','zs_close_30dmax_dif','zs_close_30dmin_dif','zs_close_30d_dif','zs_close_20_30d_avg','zs_turn_60d_avg','zs_turn_60davg_dif','zs_turn_60dmax_dif','zs_turn_60dmin_dif','zs_turn_30_60d_avg','zs_close_60davg_dif','zs_close_60dmax_dif','zs_close_60dmin_dif','zs_close_60d_dif','zs_close_30_60d_avg','zs_turn_120d_avg','zs_turn_120davg_dif','zs_turn_120dmax_dif','zs_turn_120dmin_dif','zs_turn_60_120d_avg','zs_close_120davg_dif','zs_close_120dmax_dif','zs_close_120dmin_dif','zs_close_120d_dif','zs_close_60_120d_avg','zs_turn_240d_avg','zs_turn_240davg_dif','zs_turn_240dmax_dif','zs_turn_240dmin_dif','zs_turn_120_240d_avg','zs_close_240davg_dif','zs_close_240dmax_dif','zs_close_240dmin_dif','zs_close_240d_dif','zs_close_120_240d_avg']
		all_data_raw = pd.merge(all_data_raw, zs_data, how="left", left_on=["date"],right_on=["date"])

		feature_all = pd.merge(feature_all, all_data_raw, how="left", left_on=["date",'code_market'],right_on=["date",'code_market'])

		# feature_all = feature_all.round({'open_ratio ': 5, 'close_ratio ': 5, 'high_ratio ': 5, 'low_ratio ': 5, 'pctChg ': 5, 'open_ratio_7d_avg ': 5, 'close_ratio_7d_avg ': 5, 'high_ratio_7d_avg ': 5, 'low_ratio_7d_avg ': 5, 'rsv_5 ': 5, 'k_value_5 ': 5, 'd_value_5 ': 5, 'j_value_5 ': 5, 'k_value_trend_5 ': 5, 'rsv_9 ': 5, 'k_value_9 ': 5, 'd_value_9 ': 5, 'j_value_9 ': 5, 'k_value_trend_9 ': 5, 'rsv_19 ': 5, 'k_value_19 ': 5, 'd_value_19 ': 5, 'j_value_19 ': 5, 'k_value_trend_19 ': 5, 'rsv_73 ': 5, 'k_value_73 ': 5, 'd_value_73 ': 5, 'j_value_73 ': 5, 'k_value_trend_73 ': 5, 'macd_dif_ratio ': 5, 'macd_positive_ratio_2 ': 5, 'macd_positive_ratio_3 ': 5, 'macd_positive_ratio_5 ': 5, 'macd_positive_ratio_10 ': 5, 'macd_positive_ratio_20 ': 5, 'macd_positive_ratio_40 ': 5, 'width_20 ': 5, 'close_mb20_diff ': 5, 'rsi_3d ': 5, 'rsi_5d ': 5, 'rsi_10d ': 5, 'rsi_20d ': 5, 'rsi_40d ': 5, 'industry_pctChg ': 5, 'rise_ratio ': 5, 'industry_open_ratio ': 5, 'industry_close_ratio ': 5, 'industry_high_ratio ': 5, 'industry_low_ratio ': 5, 'industry_open_ratio_7d_avg ': 5, 'industry_close_ratio_7d_avg ': 5, 'industry_high_ratio_7d_avg ': 5, 'industry_low_ratio_7d_avg ': 5, 'industry_rsv_5 ': 5, 'industry_k_value_5 ': 5, 'industry_d_value_5 ': 5, 'industry_j_value_5 ': 5, 'industry_k_value_trend_5 ': 5, 'industry_rsv_9 ': 5, 'industry_k_value_9 ': 5, 'industry_d_value_9 ': 5, 'industry_j_value_9 ': 5, 'industry_k_value_trend_9 ': 5, 'industry_rsv_19 ': 5, 'industry_k_value_19 ': 5, 'industry_d_value_19 ': 5, 'industry_j_value_19 ': 5, 'industry_k_value_trend_19 ': 5, 'industry_rsv_73 ': 5, 'industry_k_value_73 ': 5, 'industry_d_value_73 ': 5, 'industry_j_value_73 ': 5, 'industry_k_value_trend_73 ': 5, 'industry_macd_dif_ratio ': 5, 'industry_macd_positive_ratio_2 ': 5, 'industry_macd_positive_ratio_3 ': 5, 'industry_macd_positive_ratio_5 ': 5, 'industry_macd_positive_ratio_10 ': 5, 'industry_macd_positive_ratio_20 ': 5, 'industry_macd_positive_ratio_40 ': 5, 'industry_width_20 ': 5, 'industry_close_mb20_diff ': 5, 'industry_rsi_3d ': 5, 'industry_rsi_5d ': 5, 'industry_rsi_10d ': 5, 'industry_rsi_20d ': 5, 'industry_rsi_40d ': 5, 'all_pctChg ': 5, 'all_turn ': 5, 'all_rise_ratio ': 5, 'all_pctChg_3 ': 5, 'all_turn_3 ': 5, 'all_rise_ratio_3 ': 5, 'all_pctChg_5 ': 5, 'all_turn_5 ': 5, 'all_rise_ratio_5 ': 5, 'all_pctChg_10 ': 5, 'all_turn_10 ': 5, 'all_rise_ratio_10 ': 5, 'zs_open_ratio ': 5, 'zs_close_ratio ': 5, 'zs_high_ratio ': 5, 'zs_low_ratio ': 5, 'zs_pctChg ': 5, 'zs_open_ratio_7d_avg ': 5, 'zs_close_ratio_7d_avg ': 5, 'zs_high_ratio_7d_avg ': 5, 'zs_low_ratio_7d_avg ': 5, 'zs_rsv_5 ': 5, 'zs_k_value_5 ': 5, 'zs_d_value_5 ': 5, 'zs_j_value_5 ': 5, 'zs_k_value_trend_5 ': 5, 'zs_rsv_9 ': 5, 'zs_k_value_9 ': 5, 'zs_d_value_9 ': 5, 'zs_j_value_9 ': 5, 'zs_k_value_trend_9 ': 5, 'zs_rsv_19 ': 5, 'zs_k_value_19 ': 5, 'zs_d_value_19 ': 5, 'zs_j_value_19 ': 5, 'zs_k_value_trend_19 ': 5, 'zs_rsv_73 ': 5, 'zs_k_value_73 ': 5, 'zs_d_value_73 ': 5, 'zs_j_value_73 ': 5, 'zs_k_value_trend_73 ': 5, 'zs_macd_dif_ratio ': 5, 'zs_macd_positive_ratio_2 ': 5, 'zs_macd_positive_ratio_3 ': 5, 'zs_macd_positive_ratio_5 ': 5, 'zs_macd_positive_ratio_10 ': 5, 'zs_macd_positive_ratio_20 ': 5, 'zs_macd_positive_ratio_40 ': 5, 'zs_width_20 ': 5, 'zs_close_mb20_diff ': 5, 'zs_rsi_3d ': 5, 'zs_rsi_5d ': 5, 'zs_rsi_10d ': 5, 'zs_rsi_20d ': 5, 'zs_rsi_40d ': 5, 'label_7 ': 5, 'label_7_real ': 5, 'label_7_weight ': 5, 'label_7_max ': 5, 'label_7_max_real ': 5, 'label_7_max_weight ': 5, 'label_15 ': 5, 'label_15_real ': 5, 'label_15_weight ': 5, 'label_15_max ': 5, 'label_15_max_real ': 5, 'label_15_max_weight ': 5})

		return feature_all
if __name__ == '__main__':
	years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
	# years = [2008]
	for year in years:
		path = 'E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_'
		quater_path = 'E:/pythonProject/future/data/datafile/code_quarter_data_v2_all.csv'
		output_path = 'E:/pythonProject/future/data/datafile/feature/{year}_feature_v6.csv'.format(year=str(year))
		feature = Feature(path, year, quater_path)
		if os.path.isfile(output_path):
			os.remove(output_path)
		feature_all = feature.feature_process()
		feature_all.to_csv(output_path, mode='w', header=True, index=False)