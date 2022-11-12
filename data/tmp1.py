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
		raw_k_data_tmp = raw_k_data[["date","industry_id_level3", "volume_total"]].groupby(["date","industry_id_level3"]).sum()
		raw_k_data_tmp.columns = ['industry_id_level3_volume_total']
		raw_k_data_tmp['rise_ratio'] = raw_k_data[["date","industry_id_level3", "rise"]].groupby(["date","industry_id_level3"]).mean()
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
		raw_k_data["rise_ratio"] = raw_k_data["rise_ratio"]*raw_k_data['volume_total_ratio']

		industry_id_level3_k_data = raw_k_data[["industry_id_level3","open","close","preclose","high","low","turn","date","amount","pctChg","peTTM","pcfNcfTTM","pbMRQ", 'rise_ratio']].groupby(['industry_id_level3','date']).sum()
		industry_id_level3_k_data.columns = ["industry_id_level3_open","industry_id_level3_close","industry_id_level3_preclose","industry_id_level3_high","industry_id_level3_low","industry_id_level3_turn","industry_id_level3_amount","industry_id_level3_pctChg","industry_id_level3_peTTM","industry_id_level3_pcfNcfTTM","industry_id_level3_pbMRQ", 'rise_ratio']

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