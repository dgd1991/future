import copy
import gc

import pandas as pd

from feature.feature_process import *


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
		raw_k_data = raw_k_data[raw_k_data['tradestatus'] == 1]
		raw_k_data["open"] = pd.to_numeric(raw_k_data["open"], errors='coerce')
		raw_k_data["close"] = pd.to_numeric(raw_k_data["close"], errors='coerce')
		raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')
		raw_k_data["preclose"] = pd.to_numeric(raw_k_data["preclose"], errors='coerce')
		raw_k_data["high"] = pd.to_numeric(raw_k_data["high"], errors='coerce')
		raw_k_data["low"] = pd.to_numeric(raw_k_data["low"], errors='coerce')
		raw_k_data["turn"] = pd.to_numeric(raw_k_data["turn"], errors='coerce')
		raw_k_data['date'] = pd.to_datetime(raw_k_data['date'])
		raw_k_data['open_ratio'] = ((raw_k_data['open'] - raw_k_data['preclose']) / raw_k_data['preclose'])
		raw_k_data['close_ratio'] = ((raw_k_data['close'] - raw_k_data['open']) / raw_k_data['open'])
		raw_k_data['high_ratio'] = ((raw_k_data['high'] - raw_k_data['preclose']) / raw_k_data['preclose'])
		raw_k_data['low_ratio'] = ((raw_k_data['low'] - raw_k_data['preclose']) / raw_k_data['preclose'])
		raw_k_data['amount'] = raw_k_data['amount'].map(lambda x: None if x == '' else float2Bucket(float(x) * 0.00000005, 1, 0, 40000, 40000))
		raw_k_data['pctChg'] = raw_k_data['pctChg'].map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))
		raw_k_data['peTTM'] = raw_k_data['peTTM'].map(lambda x: float2Bucket(float(x), 2, 0, 2000, 4000))
		raw_k_data['pcfNcfTTM'] = raw_k_data['pcfNcfTTM'].map(lambda x: float2Bucket(float(x), 10, 0, 100, 1000))
		raw_k_data['pbMRQ'] = raw_k_data['pbMRQ'].map(lambda x: float2Bucket(float(x), 10, 0, 500, 5000))
		raw_k_data['isST'] = raw_k_data['isST'].map(lambda x: float2Bucket(float(x), 1, 0, 3, 3))

		raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))

		feature_all = copy.deepcopy(raw_k_data[['industry']])
		feature_all['open_ratio'] = raw_k_data['open_ratio'].map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))
		feature_all['open_ratio_7d_avg'] = raw_k_data.groupby(level=0)['open_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		feature_all['open_ratio_7d_avg'] = feature_all['open_ratio_7d_avg'].map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))

		feature_all['close_ratio'] = raw_k_data['close_ratio'].map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))
		feature_all['close_ratio_7d_avg'] = raw_k_data.groupby(level=0)['close_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		feature_all['close_ratio_7d_avg'] = feature_all['close_ratio_7d_avg'].map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))

		feature_all['high_ratio'] = raw_k_data['high_ratio'].map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))
		feature_all['high_ratio_7d_avg'] = raw_k_data.groupby(level=0)['high_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		feature_all['high_ratio_7d_avg'] = feature_all['high_ratio_7d_avg'].map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))

		feature_all['low_ratio'] = raw_k_data['low_ratio'].map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))
		feature_all['low_ratio_7d_avg'] = raw_k_data.groupby(level=0)['low_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean())
		feature_all['low_ratio_7d_avg'] = feature_all['low_ratio_7d_avg'].map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))

		feature_all['amount'] = raw_k_data['amount'].map(lambda x: None if x == '' else float2Bucket(float(x) * 0.00000005, 1, 0, 40000, 40000))
		feature_all['pctChg'] = raw_k_data['pctChg'].map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))
		feature_all['peTTM'] = raw_k_data['peTTM'].map(lambda x: float2Bucket(float(x), 2, 0, 2000, 4000))
		feature_all['pcfNcfTTM'] = raw_k_data['pcfNcfTTM'].map(lambda x: float2Bucket(float(x), 10, 0, 100, 1000))
		feature_all['pbMRQ'] = raw_k_data['pbMRQ'].map(lambda x: float2Bucket(float(x), 10, 0, 500, 5000))
		feature_all['isST'] = raw_k_data['isST'].map(lambda x: float2Bucket(float(x), 1, 0, 3, 3))

		feature_all['turn'] = raw_k_data['turn'].map(lambda x: float2Bucket(float(x), 2000, 0, 1, 2000))
		for i in range(20):
			if i<10:
				day_cnt = (i+1) * 5
			else:
				day_cnt = (i + 1) * 20 + 50
			feature_all['turn_' + str(day_cnt) + 'd' + '_avg'] = raw_k_data.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'] = raw_k_data['turn']/feature_all['turn_' + str(day_cnt) + 'd' + '_avg']
			feature_all['turn_' + str(day_cnt) + 'd' + '_max'] = raw_k_data.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif'] = raw_k_data['turn']/feature_all['turn_' + str(day_cnt) + 'd' + '_max']
			feature_all['turn_' + str(day_cnt) + 'd' + '_min'] = raw_k_data.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif'] = raw_k_data['turn']/feature_all['turn_' + str(day_cnt) + 'd' + '_min']
			feature_all['turn_' + str(day_cnt) + 'd' + '_avg_ratio'] = feature_all.groupby(level=0)['turn_' + str(day_cnt) + 'd' + '_avg'].apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: 1.0*(y[1])/y[0]))
			feature_all['turn_' + str(day_cnt) + 'd' + '_avg_ratio_3d'] = feature_all.groupby(level=0)['turn_' + str(day_cnt) + 'd' + '_avg'].apply(lambda x: x.rolling(min_periods=4, window=4, center=False).apply(lambda y: 1.0*(y[3])/y[0]))

			raw_k_data['close_' + str(day_cnt) + 'd' + '_avg'] = raw_k_data.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
			feature_all['close_' + str(day_cnt) + 'd' + 'avg_dif'] = raw_k_data['close']/raw_k_data['close_' + str(day_cnt) + 'd' + '_avg']
			feature_all['close_' + str(day_cnt) + 'd' + 'max_dif'] = raw_k_data['close']/raw_k_data.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
			feature_all['close_' + str(day_cnt) + 'd' + 'min_dif'] = raw_k_data['close']/raw_k_data.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
			feature_all['close_' + str(day_cnt) + 'd' + '_avg_ratio'] = raw_k_data.groupby(level=0)['close_' + str(day_cnt) + 'd' + '_avg'].apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: 1.0*(y[1])/y[0]))
			feature_all['close_' + str(day_cnt) + 'd' + '_avg_ratio_3d'] = raw_k_data.groupby(level=0)['close_' + str(day_cnt) + 'd' + '_avg'].apply(lambda x: x.rolling(min_periods=4, window=4, center=False).apply(lambda y: 1.0*(y[3])/y[0]))

			feature_all['turn_' + str(day_cnt) + 'd' + '_avg'] = feature_all['turn_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 1000, 0, 1, 1000))
			feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 200, 0, 10, 2000))
			feature_all['turn_' + str(day_cnt) + 'd' + '_max'] = feature_all['turn_' + str(day_cnt) + 'd' + '_max'].map(lambda x: float2Bucket(float(x), 1000, 0, 1, 1000))
			feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 200, 0, 10, 2000))
			feature_all['turn_' + str(day_cnt) + 'd' + '_min'] = feature_all['turn_' + str(day_cnt) + 'd' + '_min'].map(lambda x: float2Bucket(float(x), 1000, 0, 1, 1000))
			feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 200, 0, 10, 2000))
			feature_all['turn_' + str(day_cnt) + 'd' + '_avg_ratio'] = feature_all['turn_' + str(day_cnt) + 'd' + '_avg_ratio'].map(lambda x: float2Bucket(float(x), 1000, 0.5, 1.5, 1000))
			feature_all['turn_' + str(day_cnt) + 'd' + '_avg_ratio_3d'] = feature_all['turn_' + str(day_cnt) + 'd' + '_avg_ratio_3d'].map(lambda x: float2Bucket(float(x), 1000, 0.5, 1.5, 1000))

			feature_all['close_' + str(day_cnt) + 'd' + 'avg_dif'] = feature_all['close_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 1000, 0.5, 1.5, 1000))
			feature_all['close_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['close_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 200, 0, 1, 200))
			feature_all['close_' + str(day_cnt) + 'd' + 'min_dif'] = feature_all['close_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 100, 1, 20, 1000))
			feature_all['close_' + str(day_cnt) + 'd' + '_avg_ratio'] = feature_all['close_' + str(day_cnt) + 'd' + '_avg_ratio'].map(lambda x: float2Bucket(float(x), 1000, 0.5, 1.5, 1000))
			feature_all['close_' + str(day_cnt) + 'd' + '_avg_ratio_3d'] = feature_all['close_' + str(day_cnt) + 'd' + '_avg_ratio_3d'].map(lambda x: float2Bucket(float(x), 1000, 0.5, 1.5, 1000))

		feature_all = feature_all.reset_index(level=0, drop=False)
		feature_all = feature_all.reset_index(level=0, drop=False)
		feature_all = feature_all.sort_values(['date', 'code'])
		feature_all = feature_all[feature_all['date'] > str(self.year)]
		code_date = copy.deepcopy(feature_all[["date", "code"]])
		del raw_k_data
		gc.collect()

		# 季度财报特征
		profit_data = pd.read_csv(self.quarter_file_path)
		profit_data = profit_data[(profit_data['pubDate']>str(self.year - 1)) & (profit_data['pubDate']<str(self.year + 1))]
		quarter_data = copy.deepcopy(profit_data[['code', 'pubDate', 'quarter','pub_quarter','last_pub_quarter','industry']])
		quarter_data['roeAvg'] = profit_data['roeAvg'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 2.5, 400, 0, 5, 2000))
		quarter_data['npMargin'] = profit_data['npMargin'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 2.5, 400, 0, 5, 2000))
		quarter_data['gpMargin'] = profit_data['gpMargin'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 2.5, 400, 0, 5, 2000))
		quarter_data['netProfit'] = profit_data['netProfit'].map(
			lambda x: None if x == '' else bignumber2Bucket(float(x) * 0.000001, 1.03, 800, True))
		quarter_data['epsTTM'] = profit_data['epsTTM'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 30, 10, 0, 100, 1000))
		quarter_data['MBRevenue'] = profit_data['MBRevenue'].map(
			lambda x: None if x == '' else bignumber2Bucket(float(x) * 0.0000001, 1.03, 800, False))
		quarter_data['totalShare'] = profit_data['totalShare'].map(
			lambda x: None if x == '' else bignumber2Bucket(float(x) * 0.0001, 1.03, 800, False))
		quarter_data['liqaShare'] = profit_data['liqaShare'].map(
			lambda x: None if x == '' else bignumber2Bucket(float(x) * 0.0001, 1.03, 800, False))

		quarter_data['NRTurnRatio'] = profit_data['NRTurnRatio'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 10, 0, 100, 1000))
		quarter_data['INVTurnRatio'] = profit_data['INVTurnRatio'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 10, 0, 100, 1000))
		quarter_data['CATurnRatio'] = profit_data['CATurnRatio'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 5, 500))
		quarter_data['AssetTurnRatio'] = profit_data['AssetTurnRatio'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 5, 500))

		quarter_data['YOYEquity'] = profit_data['YOYEquity'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 1, 500, 0, 2, 1000))
		quarter_data['YOYAsset'] = profit_data['YOYAsset'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 1, 300, 0, 3, 900))
		quarter_data['YOYNI'] = profit_data['YOYNI'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 10, 100, 0, 20, 2000))
		quarter_data['YOYEPSBasic'] = profit_data['YOYEPSBasic'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 10, 100, 0, 20, 2000))
		quarter_data['YOYPNI'] = profit_data['YOYPNI'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 10, 100, 0, 20, 2000))

		quarter_data['currentRatio'] = profit_data['currentRatio'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 20, 2000))
		quarter_data['quickRatio'] = profit_data['quickRatio'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 20, 2000))
		quarter_data['cashRatio'] = profit_data['cashRatio'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 20, 2000))
		quarter_data['YOYLiability'] = profit_data['YOYLiability'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 5, 100, 0, 10, 1000))
		quarter_data['liabilityToAsset'] = profit_data['liabilityToAsset'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 10, 1000))
		quarter_data['assetToEquity'] = profit_data['assetToEquity'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 20, 2000))

		quarter_data['CAToAsset'] = profit_data['CAToAsset'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 1, 100))
		quarter_data['tangibleAssetToAsset'] = profit_data['tangibleAssetToAsset'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 1, 100))
		quarter_data['ebitToInterest'] = profit_data['ebitToInterest'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 100, 10, 0, 300, 3000))
		quarter_data['CFOToOR'] = profit_data['CFOToOR'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 5, 100, 0, 10, 1000))
		quarter_data['CFOToNP'] = profit_data['CFOToNP'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 25, 20, 0, 50, 1000))
		quarter_data['CFOToGr'] = profit_data['CFOToGr'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 25, 20, 0, 50, 1000))

		quarter_data['dupontROE'] = profit_data['dupontROE'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 2, 100, 0, 4, 400))
		quarter_data['dupontAssetStoEquity'] = profit_data['dupontAssetStoEquity'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 1, 11, 1000))
		quarter_data['dupontAssetTurn'] = profit_data['dupontAssetTurn'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 2, 200))
		quarter_data['dupontPnitoni'] = profit_data['dupontPnitoni'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 1, 100))
		quarter_data['dupontNitogr'] = profit_data['dupontNitogr'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 1, 100, 0, 2, 200))
		quarter_data['dupontTaxBurden'] = profit_data['dupontTaxBurden'].map(
			lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 1, 100))
		quarter_data['dupontIntburden'] = profit_data['dupontIntburden'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 1.5, 100, 0, 3, 300))
		quarter_data['dupontEbittogr'] = profit_data['dupontEbittogr'].map(
			lambda x: None if x == '' else float2Bucket(float(x) + 1, 100, 0, 2, 200))

		quarter_data_final = pd.merge(quarter_data, quarter_data, how="left", left_on=['code', "pub_quarter"], right_on=['code', 'last_pub_quarter'])
		code_date['quarter'] = pd.to_numeric(
			code_date['date'].map(lambda x: get_date_quarter(str(x).split('-')[0], str(x).split('-')[1], False)), errors='coerce')
		quarter_data_final["pub_quarter_x"] = pd.to_numeric(quarter_data_final["pub_quarter_x"], errors='coerce')
		fea_tmp = pd.merge(code_date, quarter_data_final, how="left", left_on=['code', 'quarter'],
		                   right_on=['code', 'pub_quarter_x'])
		fea_list = ['roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare',
		            'NRTurnRatio', 'INVTurnRatio', 'CATurnRatio', 'AssetTurnRatio', 'YOYEquity', 'YOYAsset', 'YOYNI',
		            'YOYEPSBasic', 'YOYPNI', 'currentRatio', 'quickRatio', 'cashRatio', 'YOYLiability',
		            'liabilityToAsset', 'assetToEquity', 'CAToAsset', 'tangibleAssetToAsset', 'ebitToInterest',
		            'CFOToOR', 'CFOToNP', 'CFOToGr', 'dupontROE', 'dupontAssetStoEquity', 'dupontAssetTurn',
		            'dupontPnitoni', 'dupontNitogr', 'dupontTaxBurden', 'dupontIntburden', 'dupontEbittogr']
		del quarter_data
		del quarter_data_final
		gc.collect()
		for fea in fea_list:
			print(fea)
			feature_all[fea] = fea_tmp[['date', 'pubDate_x', fea + '_x', fea + '_y']].apply(
				lambda x: x[fea + '_x'] if str(x.date) > str(x.pubDate_x) else x[fea + '_y'], axis=1)

		# 行业特征特征
		raw_k_data = pd.read_csv(self.k_file_path)
		raw_k_data_his = pd.read_csv(self.k_file_path_his)
		raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
		del raw_k_data_his
		gc.collect()
		raw_k_data["tradestatus"] = pd.to_numeric(raw_k_data["tradestatus"], errors='coerce')
		raw_k_data = raw_k_data[raw_k_data['tradestatus'] == 1]
		raw_k_data = raw_k_data[["open","industry","close","high","low","turn","date","amount","pctChg","peTTM","pcfNcfTTM","pbMRQ"]]
		raw_k_data["open"] = pd.to_numeric(raw_k_data["open"], errors='coerce')
		raw_k_data["industry"] = pd.to_numeric(raw_k_data["industry"], errors='coerce')
		raw_k_data["close"] = pd.to_numeric(raw_k_data["close"], errors='coerce')
		raw_k_data["high"] = pd.to_numeric(raw_k_data["high"], errors='coerce')
		raw_k_data["low"] = pd.to_numeric(raw_k_data["low"], errors='coerce')
		raw_k_data["turn"] = pd.to_numeric(raw_k_data["turn"], errors='coerce')
		raw_k_data['date'] = pd.to_datetime(raw_k_data['date'])
		raw_k_data["amount"] = pd.to_numeric(raw_k_data["amount"], errors='coerce')
		raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')
		raw_k_data["peTTM"] = pd.to_numeric(raw_k_data["peTTM"], errors='coerce')
		raw_k_data["pcfNcfTTM"] = pd.to_numeric(raw_k_data["pcfNcfTTM"], errors='coerce')
		raw_k_data["pbMRQ"] = pd.to_numeric(raw_k_data["pbMRQ"], errors='coerce')
		raw_k_data = raw_k_data.groupby(['industry', 'date']).mean()


		raw_k_data = raw_k_data.groupby(['date', 'industry'])


		return feature_all
if __name__ == '__main__':
	years = [2010, 2011, 2012, 2013, 2014, 2015, 2016]
	for year in years:
		path = 'E:/pythonProject/future/data/datafile/code_k_data_v2_'
		quater_path = 'E:/pythonProject/future/data/datafile/code_quarter_data_v2_all.csv'
		feature = Feature(path, year, quater_path)
		feature_all = feature.feature_process()
		feature_all.to_csv('E:/pythonProject/future/data/datafile/feature/{year}_feature.csv'.format(year=str(year)),
		                 mode='w', header=True, index=False)

