import copy
import gc
from itertools import zip_longest

import pandas as pd

from feature.feature_process import *
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
		raw_k_data['pctChg'] = raw_k_data['pctChg']
		raw_k_data['peTTM'] = raw_k_data['peTTM']
		raw_k_data['pcfNcfTTM'] = raw_k_data['pcfNcfTTM']
		raw_k_data['pbMRQ'] = raw_k_data['pbMRQ']
		raw_k_data['isST'] = raw_k_data['isST']

		raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))

		feature_all = copy.deepcopy(raw_k_data[['industry','open_ratio','close_ratio','high_ratio','low_ratio']])
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

		# 行业特征
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

		industry_k_data = raw_k_data[["industry","open","close","preclose","high","low","turn","date","amount","pctChg","peTTM","pcfNcfTTM","pbMRQ", 'rise_ratio']].groupby(['industry','date']).sum()
		industry_k_data.columns = ["industry_open","industry_close","industry_preclose","industry_high","industry_low","industry_turn","industry_amount","industry_pctChg","industry_peTTM","industry_pcfNcfTTM","industry_pbMRQ", 'rise_ratio']

		industry_k_data['industry_open_ratio'] = ((industry_k_data["industry_open"] - industry_k_data["industry_preclose"]) / industry_k_data["industry_preclose"])
		industry_k_data['industry_close_ratio'] = ((industry_k_data["industry_close"] - industry_k_data["industry_open"]) / industry_k_data["industry_open"])
		industry_k_data['industry_high_ratio'] = ((industry_k_data["industry_high"] - industry_k_data["industry_preclose"]) / industry_k_data["industry_preclose"])
		industry_k_data['industry_low_ratio'] = ((industry_k_data["industry_low"] - industry_k_data["industry_preclose"]) / industry_k_data["industry_preclose"])

		industry_k_data['industry_open_ratio_7d_avg'] = industry_k_data['industry_open_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_k_data['industry_close_ratio_7d_avg'] = industry_k_data['industry_close_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_k_data['industry_high_ratio_7d_avg'] = industry_k_data['industry_high_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		industry_k_data['industry_low_ratio_7d_avg'] = industry_k_data['industry_low_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())

		industry_k_data['industry_turn'] = industry_k_data["industry_turn"].map(lambda x: float2Bucket(float(x), 1, 0, 200, 200))
		industry_k_data['industry_amount'] = industry_k_data["industry_amount"].map(lambda x: None if x == '' else float2Bucket(float(x) * 0.00000001, 1, 0, 10000, 10000))
		industry_k_data['industry_peTTM'] = industry_k_data["industry_peTTM"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 1500, 1500))
		industry_k_data['industry_pcfNcfTTM'] = industry_k_data["industry_pcfNcfTTM"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 2000, 2000))
		industry_k_data['industry_pbMRQ'] = industry_k_data["industry_pbMRQ"].map(lambda x: float2Bucket(float(x) + 500, 1, 0, 2000, 2000))


		# kdj 5, 9, 19, 36, 45, 73，
		# 任意初始化，超过30天后的kdj值基本一样
		tmp = industry_k_data[["industry_low", "industry_high", "industry_close"]]
		for day_cnt in [5, 9, 19, 73]:
		# industry_k_data['industry_rsv'] = industry_k_data[["industry_low", "industry_high", "industry_close"]].groupby(level=0).apply(lambda x: (x.close-x.low.rolling(min_periods=1, window=day_cnt, center=False).min())/(x.high.rolling(min_periods=1, window=day_cnt, center=False).max()-x.low.rolling(min_periods=1, window=day_cnt, center=False).min()))
			tmp['min'] = tmp["industry_low"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			tmp['max'] = tmp["industry_high"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_k_data['industry_rsv_' + str(day_cnt)] = (tmp["industry_close"] - tmp['min'])/(tmp['max'] - tmp['min'])
			industry_k_data['industry_k_value_' + str(day_cnt)] = industry_k_data['industry_rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
			industry_k_data['industry_d_value_' + str(day_cnt)] = industry_k_data['industry_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
			industry_k_data['industry_j_value_' + str(day_cnt)] = 3 * industry_k_data['industry_k_value_' + str(day_cnt)] - 2 * industry_k_data['industry_d_value_' + str(day_cnt)]
			industry_k_data['industry_k_value_trend_' + str(day_cnt)] = industry_k_data['industry_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1])-y[0]))
			industry_k_data['industry_kd_value' + str(day_cnt)] = (industry_k_data['industry_k_value_' + str(day_cnt)] - industry_k_data['industry_d_value_' + str(day_cnt)]).rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1))

		# macd 12
		# 任意初始化，超过30天后的macd值基本一样
		tmp = industry_k_data[['industry_close']]
		tmp['ema_12'] = tmp['industry_close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 13, adjust=False).mean())
		tmp['ema_26'] = tmp['industry_close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 27, adjust=False).mean())
		tmp['macd_dif'] = tmp['ema_12'] - tmp['ema_26']
		tmp['macd_dea'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0 / 5, adjust=False).mean())
		tmp['macd_dif_max'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
		tmp['macd_dif_min'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
		tmp['macd_dea_max'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
		tmp['macd_dea_min'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
		tmp['macd'] = (tmp['macd_dif'] - tmp['macd_dea']) * 2
		industry_k_data['industry_macd_positive'] = tmp['macd'].apply(lambda x: 1 if x>0 else 0)
		industry_k_data['industry_macd_dif_ratio'] = (tmp['macd_dif']-tmp['macd_dif_min'])/(tmp['macd_dif_max']-tmp['macd_dif_min'])
		for day_cnt in [2, 3, 5, 10, 20, 40]:
			industry_k_data['industry_macd_dif_' + str(day_cnt)] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_k_data['industry_macd_dea_' + str(day_cnt)] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_k_data['industry_macd_' + str(day_cnt)] = tmp['macd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
			industry_k_data['industry_macd_positive_ratio_' + str(day_cnt)] = industry_k_data['industry_macd_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
		industry_k_data['industry_macd_dif_dea'] = (tmp['macd_dif']-tmp['macd_dea']).groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1)))

		# boll线
		tmp['mb_20'] = tmp["industry_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).mean())
		tmp['md_20'] = tmp["industry_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).std())
		tmp['up_20'] = tmp['mb_20'] + 2 * tmp['md_20']
		tmp['dn_20'] = tmp['mb_20'] - 2 * tmp['md_20']
		industry_k_data['industry_width_20'] = 4 * tmp['md_20'] / tmp['mb_20']
		industry_k_data['industry_close_mb20_diff'] = (tmp["industry_close"] - tmp['mb_20'])/(2 * tmp['md_20'])

		# cr指标
		tmp = industry_k_data[["industry_close", "industry_open", "industry_high", "industry_low"]]
		tmp['cr_m'] = (tmp["industry_close"] + tmp["industry_open"] + tmp["industry_high"] + tmp["industry_low"])/4
		tmp['cr_ym'] = tmp['cr_m'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0]))
		tmp['cr_p1_day'] = (tmp["industry_high"] - tmp['cr_ym'])
		tmp['cr_p2_day'] = (tmp['cr_ym'] - tmp["industry_low"])
		for day_cnt in (3, 5, 10, 20, 40):
			tmp['cr_p1_' + str(day_cnt) + 'd'] = tmp['cr_p1_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			tmp['cr_p2_' + str(day_cnt) + 'd'] = tmp['cr_p2_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			industry_k_data['industry_cr_' + str(day_cnt) + 'd'] = tmp['cr_p1_' + str(day_cnt) + 'd'] / tmp['cr_p2_' + str(day_cnt) + 'd']
			industry_k_data['industry_cr_' + str(day_cnt) + 'd'] = industry_k_data['industry_cr_' + str(day_cnt) + 'd'].map(lambda x: float2Bucket(float(x) * 100, 0.2, 0, 500, 100))
		# rsi指标
		tmp = industry_k_data[["industry_close", "industry_preclose"]]
		tmp['price_dif'] = tmp["industry_close"] - tmp["industry_preclose"]
		tmp['rsi_positive'] = tmp['price_dif'].apply(lambda x: max(x, 0))
		tmp['rsi_all'] = tmp['price_dif'].apply(lambda x: abs(x))
		for day_cnt in (3, 5, 10, 20, 40):
			tmp['rsi_positive_sum'] = tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			tmp['rsi_all_sum'] = tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
			industry_k_data['industry_rsi_' + str(day_cnt) + 'd'] =  tmp['rsi_positive_sum'] / tmp['rsi_all_sum']

		tmp = industry_k_data[["industry_turn", "industry_close"]]
		day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
		for index in range(len(day_cnt_list)):
			day_cnt = day_cnt_list[index]
			day_cnt_last = day_cnt_list[index-1]
			industry_k_data['industry_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_k_data["industry_turn"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_k_data["industry_turn"]/industry_k_data['industry_turn_' + str(day_cnt) + 'd' + '_avg']
			industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_k_data['industry_turn_' + str(day_cnt) + 'd' + 'avg_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			if index>0:
				industry_k_data['industry_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_k_data['industry_turn_' + str(day_cnt_last) + 'd' + '_avg']/industry_k_data['industry_turn_' + str(day_cnt) + 'd' + '_avg']

			tmp['industry_close_' + str(day_cnt) + 'd' + '_avg'] = tmp["industry_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_k_data["industry_close"]/tmp['industry_close_' + str(day_cnt) + 'd' + '_avg']
			industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_k_data["industry_close"]/industry_k_data["industry_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			industry_k_data['industry_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_k_data["industry_close"]/industry_k_data["industry_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			industry_k_data['industry_close_' + str(day_cnt) + 'd' + '_dif'] = industry_k_data["industry_close"].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 1.0*(y[day_cnt-1])/y[0]))
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

		feature_all = pd.merge(feature_all, industry_k_data, how="left", left_on=["date",'industry'],right_on=["date",'industry'])
		feature_all = feature_all.sort_values(['date', 'code'])
		gc.collect()

		return feature_all
if __name__ == '__main__':
	years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
	# years = [2008]
	for year in years:
		path = 'E:/pythonProject/stock/data/datafile/raw_feature/code_k_data_v4_'
		quater_path = 'E:/pythonProject/stock/data/datafile/code_quarter_data_v2_all.csv'
		feature = Feature(path, year, quater_path)
		feature_all = feature.feature_process()
		# print(feature_all)
		feature_all.to_csv('E:/pythonProject/stock/data/datafile/feature/{year}_feature_v4.csv'.format(year=str(year)),
		                 mode='w', header=True, index=False)

