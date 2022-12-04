      raw_k_data = pd.read_csv(self.k_file_path)
      raw_k_data_his = pd.read_csv(self.k_file_path_his)
      raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
      raw_k_data = raw_k_data[raw_k_data['industry_id_level3']>0]
      del raw_k_data_his
      gc.collect()
      raw_k_data["tradestatus"] = pd.to_numeric(raw_k_data["tradestatus"], errors='coerce')
      raw_k_data["turn"] = pd.to_numeric(raw_k_data["turn"], errors='coerce')
      raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')
      # 需要去除上市当天的股票
      raw_k_data = raw_k_data[(raw_k_data['tradestatus'] == 1) & (raw_k_data['turn'] > 0) & (raw_k_data['pctChg'] <= 20) & (raw_k_data['pctChg'] >= -20)]
      raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))
      raw_k_data['is_new'] = raw_k_data["pctChg"].groupby(level=0).apply(lambda x: x.rolling(min_periods=20, window=20, center=False).apply(lambda y: y[0]))
      raw_k_data = raw_k_data[raw_k_data['is_new'].map(lambda x: False if np.isnan(x) else True)]
      raw_k_data = raw_k_data.reset_index(level=0, drop=True)
      raw_k_data = raw_k_data.reset_index(level=0, drop=False)

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
      raw_k_data['pctChg'] = raw_k_data['pctChg'].map(lambda x: x/100.0)
      # 计算的是流通部分的市值
      raw_k_data["market_value"] = 0.00000001 * raw_k_data['amount']/(raw_k_data['turn'].map(lambda x: x/100.0))
      raw_k_data_tmp = raw_k_data[["date","industry_id_level3", "market_value"]].groupby(["industry_id_level3","date"]).sum()
      raw_k_data_tmp.columns = ['industry_id_level3_market_value']
      raw_k_data_tmp['industry_id_level3_rise_ratio'] = raw_k_data[["date","industry_id_level3", "rise"]].groupby(["industry_id_level3","date"]).mean()
      raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
      raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
      raw_k_data = pd.merge(raw_k_data, raw_k_data_tmp, how="left", left_on=["date","industry_id_level3"], right_on=["date","industry_id_level3"])
      raw_k_data['market_value_ratio'] = raw_k_data["market_value"]/raw_k_data["industry_id_level3_market_value"]

      raw_k_data["turn"] = raw_k_data["turn"]*raw_k_data['market_value_ratio']
      raw_k_data['open_ratio'] = ((raw_k_data['open'] - raw_k_data['preclose']) / raw_k_data['preclose'])*raw_k_data['market_value_ratio']
      raw_k_data['close_ratio'] = ((raw_k_data['close'] - raw_k_data['open']) / raw_k_data['open'])*raw_k_data['market_value_ratio']
      raw_k_data['high_ratio'] = ((raw_k_data['high'] - raw_k_data['preclose']) / raw_k_data['preclose'])*raw_k_data['market_value_ratio']
      raw_k_data['low_ratio'] = ((raw_k_data['low'] - raw_k_data['preclose']) / raw_k_data['preclose'])*raw_k_data['market_value_ratio']
      raw_k_data['pctChg'] = raw_k_data['pctChg']*raw_k_data['market_value_ratio']
      raw_k_data_tmp['industry_id_level3_rise_ratio'] = raw_k_data_tmp['industry_id_level3_rise_ratio']*raw_k_data['market_value_ratio']

      raw_k_data["peTTM"] = raw_k_data["peTTM"]*raw_k_data['market_value_ratio']
      raw_k_data["pcfNcfTTM"] = raw_k_data["pcfNcfTTM"]*raw_k_data['market_value_ratio']
      raw_k_data["pbMRQ"] = raw_k_data["pbMRQ"]*raw_k_data['market_value_ratio']
      raw_k_data["industry_id_level3_rise_ratio"] = raw_k_data["industry_id_level3_rise_ratio"]*raw_k_data['market_value_ratio']

      industry_id_level3_k_data = raw_k_data[["industry_id_level3","open_ratio","close_ratio","high_ratio","low_ratio","turn","date","pctChg","peTTM","pcfNcfTTM","pbMRQ", 'industry_id_level3_rise_ratio', 'market_value']].groupby(['industry_id_level3','date']).sum().round(5)
      industry_id_level3_k_data.columns = ["industry_id_level3_open_ratio","industry_id_level3_close_ratio","industry_id_level3_high_ratio","industry_id_level3_low_ratio","industry_id_level3_turn","industry_id_level3_pctChg","industry_id_level3_peTTM","industry_id_level3_pcfNcfTTM","industry_id_level3_pbMRQ", 'industry_id_level3_rise_ratio', 'industry_id_level3_market_value']
      del raw_k_data
      gc.collect()
      if os.path.isfile('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level3_' + str(year-1) + '.csv'):
          industry_id_level3_k_data_his = pd.read_csv('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level3_' + str(year-1) + '.csv')
          industry_id_level3_k_data_his = industry_id_level3_k_data_his[['industry_id_level3_close']]
      industry_id_level3_k_data = industry_id_level3_k_data.reset_index(level=0, drop=False)
      industry_id_level3_k_data = industry_id_level3_k_data.reset_index(level=0, drop=False)
      if os.path.isfile('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level3_' + str(year-1) + '.csv'):
         industry_id_level3_k_data = pd.merge(industry_id_level3_k_data, industry_id_level3_k_data_his, how="left", left_on=['industry_id_level3','date'], right_on=['industry_id_level3','date'])
      else:
         industry_id_level3_k_data['industry_id_level3_close'] = industry_id_level3_k_data['industry_id_level3_open_ratio'].apply(lambda x: 1000)

      # industry_id_level3_k_data = industry_id_level3_k_data.set_index('industry_id_level3', drop=True)
      # industry_id_level3_k_data = industry_id_level3_k_data.set_index('date', drop=True)
      industry_id_level3_k_data = industry_id_level3_k_data.groupby('industry_id_level3').apply(lambda x: x.set_index('date', drop=True))

      industry_id_level3_k_data = industry_id_level3_k_data.groupby(level=0).apply(lambda x: self.func(x, 'industry_id_level3_pctChg', 'industry_id_level3_close')).round(5)
      industry_id_level3_k_data['industry_id_level3_preclose'] = industry_id_level3_k_data['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0])).round(5)
      industry_id_level3_k_data['industry_id_level3_open'] = industry_id_level3_k_data['industry_id_level3_preclose']*(industry_id_level3_k_data['industry_id_level3_open_ratio'].apply(lambda x: x + 1)).round(5)
      industry_id_level3_k_data['industry_id_level3_high'] = industry_id_level3_k_data['industry_id_level3_preclose']*(industry_id_level3_k_data['industry_id_level3_high_ratio'].apply(lambda x: x + 1)).round(5)
      industry_id_level3_k_data['industry_id_level3_low'] = industry_id_level3_k_data['industry_id_level3_preclose']*(industry_id_level3_k_data['industry_id_level3_low_ratio'].apply(lambda x: x + 1)).round(5)

      # 写出指数点数
      industry_id_level3_k_data[['industry_id_level3_open', 'industry_id_level3_close', 'industry_id_level3_high', 'industry_id_level3_low']].to_csv('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level3_' + str(year) + '.csv', mode='w', header=True, index=False)

      industry_id_level3_k_data['industry_id_level3_open_ratio_7d_avg'] = industry_id_level3_k_data['industry_id_level3_open_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level3_k_data['industry_id_level3_close_ratio_7d_avg'] = industry_id_level3_k_data['industry_id_level3_close_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level3_k_data['industry_id_level3_high_ratio_7d_avg'] = industry_id_level3_k_data['industry_id_level3_high_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level3_k_data['industry_id_level3_low_ratio_7d_avg'] = industry_id_level3_k_data['industry_id_level3_low_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level3_k_data['industry_id_level3_turn'] = industry_id_level3_k_data["industry_id_level3_turn"].map(lambda x: float2Bucket(float(x) * 100, 1, 0, 50, 50))

      industry_id_level3_k_data['industry_id_level3_peTTM'] = industry_id_level3_k_data["industry_id_level3_peTTM"].map(lambda x: float2Bucket(float(x) + 200, 0.5, 0, 400, 200))
      industry_id_level3_k_data['industry_id_level3_pcfNcfTTM'] = industry_id_level3_k_data["industry_id_level3_pcfNcfTTM"].map(lambda x: float2Bucket(float(x) + 200, 0.5, 0, 400, 200))
      industry_id_level3_k_data['industry_id_level3_pbMRQ'] = industry_id_level3_k_data["industry_id_level3_pbMRQ"].map(lambda x: float2Bucket(float(x), 2, 0, 100, 200))
      industry_id_level3_k_data['industry_id_level3_market_value'] = industry_id_level3_k_data["industry_id_level3_market_value"].map(lambda x: None if x == '' else bignumber2Bucket(float(x), 1.25, 60))

      # kdj 5, 9, 19, 36, 45, 73，
      # 任意初始化，超过30天后的kdj值基本一样
      tmp = industry_id_level3_k_data[["industry_id_level3_low", "industry_id_level3_high", "industry_id_level3_close"]]
      for day_cnt in [5, 9, 19, 73]:
      # industry_id_level3_k_data['industry_id_level3_rsv'] = industry_id_level3_k_data[["industry_id_level3_low", "industry_id_level3_high", "industry_id_level3_close"]].groupby(level=0).apply(lambda x: (x.close-x.low.rolling(min_periods=1, window=day_cnt, center=False).min())/(x.high.rolling(min_periods=1, window=day_cnt, center=False).max()-x.low.rolling(min_periods=1, window=day_cnt, center=False).min()))
         tmp['min'] = tmp["industry_id_level3_low"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
         tmp['max'] = tmp["industry_id_level3_high"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
         industry_id_level3_k_data['industry_id_level3_rsv_' + str(day_cnt)] = ((tmp["industry_id_level3_close"] - tmp['min'])/(tmp['max'] - tmp['min'])).round(5)
         industry_id_level3_k_data['industry_id_level3_k_value_' + str(day_cnt)] = industry_id_level3_k_data['industry_id_level3_rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean()).round(5)
         industry_id_level3_k_data['industry_id_level3_d_value_' + str(day_cnt)] = industry_id_level3_k_data['industry_id_level3_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean()).round(5)
         industry_id_level3_k_data['industry_id_level3_j_value_' + str(day_cnt)] = (3 * industry_id_level3_k_data['industry_id_level3_k_value_' + str(day_cnt)] - 2 * industry_id_level3_k_data['industry_id_level3_d_value_' + str(day_cnt)]).round(5)
         # industry_id_level3_k_data['industry_id_level3_k_value_trend_' + str(day_cnt)] = industry_id_level3_k_data['industry_id_level3_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1])-y[0]))
         # industry_id_level3_k_data['industry_id_level3_kd_value' + str(day_cnt)] = (industry_id_level3_k_data['industry_id_level3_k_value_' + str(day_cnt)] - industry_id_level3_k_data['industry_id_level3_d_value_' + str(day_cnt)]).rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1))

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
      industry_id_level3_k_data['industry_id_level3_macd_dif_ratio'] = ((tmp['macd_dif']-tmp['macd_dif_min'])/(tmp['macd_dif_max']-tmp['macd_dif_min'])).round(5)
      for day_cnt in [2, 3, 5, 10, 20, 40]:
         industry_id_level3_k_data['industry_id_level3_macd_dif_' + str(day_cnt)] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         industry_id_level3_k_data['industry_id_level3_macd_dea_' + str(day_cnt)] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         industry_id_level3_k_data['industry_id_level3_macd_' + str(day_cnt)] = tmp['macd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         industry_id_level3_k_data['industry_id_level3_macd_positive_ratio_' + str(day_cnt)] = industry_id_level3_k_data['industry_id_level3_macd_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean()).round(5)
      industry_id_level3_k_data['industry_id_level3_macd_dif_dea'] = (tmp['macd_dif']-tmp['macd_dea']).groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1)))

      # boll线
      for day_cnt in [2, 3, 5, 10, 20, 40]:
         tmp['mb_' + str(day_cnt)] = tmp['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
         tmp['md_' + str(day_cnt)] = tmp['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).std())
         tmp['up_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] + 2 * tmp['md_' + str(day_cnt)]
         tmp['dn_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] - 2 * tmp['md_' + str(day_cnt)]
         industry_id_level3_k_data['industry_id_level3_width_' + str(day_cnt)] = (4 * tmp['md_' + str(day_cnt)] / tmp['mb_' + str(day_cnt)]).round(5)
         industry_id_level3_k_data['industry_id_level3_close_mb' + str(day_cnt) + '_diff'] = ((tmp['industry_id_level3_close'] - tmp['mb_' + str(day_cnt)])/(2 * tmp['md_' + str(day_cnt)])).round(5)

      # cr指标
      tmp = industry_id_level3_k_data[["industry_id_level3_close", "industry_id_level3_open", "industry_id_level3_high", "industry_id_level3_low"]]
      tmp['cr_m'] = (tmp["industry_id_level3_close"] + tmp["industry_id_level3_open"] + tmp["industry_id_level3_high"] + tmp["industry_id_level3_low"])/4
      tmp['cr_ym'] = tmp['cr_m'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0]))
      tmp['cr_p1_day'] = (tmp['industry_id_level3_high'] - tmp['cr_ym']).apply(lambda x: max(x, 0))
      tmp['cr_p2_day'] = (tmp['cr_ym'] - tmp['industry_id_level3_low']).apply(lambda x: max(x, 0))
      for day_cnt in [26]:
         tmp['cr_p1_' + str(day_cnt) + 'd'] = tmp['cr_p1_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         tmp['cr_p2_' + str(day_cnt) + 'd'] = tmp['cr_p2_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         tmp['cr_' + str(day_cnt) + 'd'] = tmp['cr_p1_' + str(day_cnt) + 'd'] / tmp['cr_p2_' + str(day_cnt) + 'd']
         tmp['cr_a_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=5, center=False).mean())
         tmp['cr_b_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=10, center=False).mean())
         tmp['cr_c_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).mean())
         tmp['cr_d_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=60, center=False).mean())
         industry_id_level3_k_data['industry_id_level3_cr_bias_' + str(day_cnt) + 'd'] = (tmp['cr_' + str(day_cnt) + 'd']/tmp['cr_a_' + str(day_cnt) + 'd']).round(5)
         industry_id_level3_k_data['industry_id_level3_cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: float2Bucket(float(x)*100, 0.1, 0, 300, 30))

         tmp['cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_a_' + str(day_cnt) + 'd'] = tmp['cr_a_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_b_' + str(day_cnt) + 'd'] = tmp['cr_b_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_c_' + str(day_cnt) + 'd'] = tmp['cr_c_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_d_' + str(day_cnt) + 'd'] = tmp['cr_d_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         # bucket空间可以设置成 75万，多个特征可以共享embedding
         industry_id_level3_k_data['industry_id_level3_cr_trend_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].str.cat([tmp['cr_a_' + str(day_cnt) + 'd'],tmp['cr_b_' + str(day_cnt) + 'd'],tmp['cr_c_' + str(day_cnt) + 'd'],tmp['cr_d_' + str(day_cnt) + 'd']], sep='_')
         for day_cnt_new in range(4):
            industry_id_level3_k_data = industry_id_level3_k_data.groupby(level=0).apply(lambda x: self._object_rolling(x, 'industry_id_level3_cr_trend_' + str(day_cnt) + 'd', 'industry_id_level3_cr_trend_' + str(day_cnt) + 'd' + '_' + str(day_cnt_new), day_cnt_new+2, 0))

      # rsi指标
      tmp = industry_id_level3_k_data[["industry_id_level3_close", "industry_id_level3_preclose"]]
      tmp['price_dif'] = tmp["industry_id_level3_close"] - tmp["industry_id_level3_preclose"]
      tmp['rsi_positive'] = tmp['price_dif'].apply(lambda x: max(x, 0))
      tmp['rsi_all'] = tmp['price_dif'].apply(lambda x: abs(x))
      for day_cnt in (3, 5, 10, 20, 40):
         tmp['rsi_positive_sum'] = tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         tmp['rsi_all_sum'] = tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         industry_id_level3_k_data['industry_id_level3_rsi_' + str(day_cnt) + 'd'] =  (tmp['rsi_positive_sum'] / tmp['rsi_all_sum']).round(5)

      tmp = industry_id_level3_k_data[["industry_id_level3_turn", "industry_id_level3_close"]]
      day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
      for index in range(len(day_cnt_list)):
         day_cnt = day_cnt_list[index]
         day_cnt_last = day_cnt_list[index-1]
         industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level3_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
         industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = tmp['industry_id_level3_turn']/industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg']
         industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg']/tmp['industry_id_level3_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
         industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'min_dif'] = tmp['industry_id_level3_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())/industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg']
         if index>0:
            industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + 'd' + '_avg']/industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg']
            industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + 'd' + 'max_dif']/industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'max_dif']
            industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + 'd' + 'min_dif']/industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'min_dif']

         tmp['industry_id_level3_close_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
         industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'avg_dif'] = tmp['industry_id_level3_close']/tmp['industry_id_level3_close_' + str(day_cnt) + 'd' + '_avg']
         industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'max_dif'] = tmp['industry_id_level3_close']/tmp['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
         industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'min_dif'] = tmp['industry_id_level3_close']/tmp['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
         industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + '_dif'] = tmp['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 1.0*(y[day_cnt-1])/y[0]))
         if index>0:
            industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level3_close_' + str(day_cnt_last) + 'd' + '_avg']/tmp['industry_id_level3_close_' + str(day_cnt) + 'd' + '_avg']

      for index in range(len(day_cnt_list)):
         day_cnt = day_cnt_list[index]
         day_cnt_last = day_cnt_list[index - 1]
         if index > 0:
            industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 2, 100))
            industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 2, 100))
            industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 2, 100))
            industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 4, 200))
         industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: min(2, round(x/100, 4)))
         industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
         industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: round(x, 4))
         industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level3_k_data['industry_id_level3_turn_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: round(x, 4))

         industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
         industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 100, 0, 1, 100))
         industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 1, 10, 500))
         industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + '_dif'] = industry_id_level3_k_data['industry_id_level3_close_' + str(day_cnt) + 'd' + '_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

      # 还需要再做一些量价类特征，比如高位放量，低位缩量等等
      for day_cnt in [3, 5, 10, 20, 30, 60, 120, 240]:
         industry_id_level3_k_data['industry_id_level3_max_turn_index' + str(day_cnt) + 'd'] = tmp['industry_id_level3_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmax()))
         industry_id_level3_k_data['industry_id_level3_max_close_index' + str(day_cnt) + 'd'] = tmp['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmax()))
         industry_id_level3_k_data['industry_id_level3_min_close_index' + str(day_cnt) + 'd'] = tmp['industry_id_level3_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmin()))
         industry_id_level3_k_data = industry_id_level3_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'industry_id_level3_max_turn_index' + str(day_cnt) + 'd', 'industry_id_level3_close', 'industry_id_level3_max_turn_close' + str(day_cnt) + 'd', day_cnt))
         industry_id_level3_k_data = industry_id_level3_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'industry_id_level3_max_close_index' + str(day_cnt) + 'd', 'industry_id_level3_turn', 'industry_id_level3_max_closs_turn' + str(day_cnt) + 'd', day_cnt))
         industry_id_level3_k_data = industry_id_level3_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'industry_id_level3_min_close_index' + str(day_cnt) + 'd', 'industry_id_level3_turn', 'industry_id_level3_min_closs_turn' + str(day_cnt) + 'd', day_cnt))

         industry_id_level3_k_data['industry_id_level3_max_turn_close' + str(day_cnt) + 'd'] = (tmp['industry_id_level3_close']/industry_id_level3_k_data['industry_id_level3_max_turn_close' + str(day_cnt) + 'd']).map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
         industry_id_level3_k_data['industry_id_level3_max_closs_turn' + str(day_cnt) + 'd'] = (industry_id_level3_k_data['industry_id_level3_max_closs_turn' + str(day_cnt) + 'd']/tmp['industry_id_level3_turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
         industry_id_level3_k_data['industry_id_level3_min_closs_turn' + str(day_cnt) + 'd'] = (industry_id_level3_k_data['industry_id_level3_min_closs_turn' + str(day_cnt) + 'd']/tmp['industry_id_level3_turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))

      industry_id_level3_k_data = industry_id_level3_k_data.reset_index(level=0, drop=True)
      industry_id_level3_k_data = industry_id_level3_k_data.reset_index(level=0, drop=False)
      selected_columns = ['date','industry_id_level3','industry_id_level3_open_ratio','industry_id_level3_close_ratio','industry_id_level3_high_ratio','industry_id_level3_low_ratio','industry_id_level3_turn','industry_id_level3_pctChg','industry_id_level3_peTTM','industry_id_level3_pcfNcfTTM','industry_id_level3_pbMRQ','industry_id_level3_rise_ratio','industry_id_level3_market_value','industry_id_level3_open_ratio_7d_avg','industry_id_level3_close_ratio_7d_avg','industry_id_level3_high_ratio_7d_avg','industry_id_level3_low_ratio_7d_avg','industry_id_level3_rsv_5','industry_id_level3_k_value_5','industry_id_level3_d_value_5','industry_id_level3_j_value_5','industry_id_level3_rsv_9','industry_id_level3_k_value_9','industry_id_level3_d_value_9','industry_id_level3_j_value_9','industry_id_level3_rsv_19','industry_id_level3_k_value_19','industry_id_level3_d_value_19','industry_id_level3_j_value_19','industry_id_level3_rsv_73','industry_id_level3_k_value_73','industry_id_level3_d_value_73','industry_id_level3_j_value_73','industry_id_level3_macd_positive','industry_id_level3_macd_dif_ratio','industry_id_level3_macd_dif_2','industry_id_level3_macd_dea_2','industry_id_level3_macd_2','industry_id_level3_macd_positive_ratio_2','industry_id_level3_macd_dif_3','industry_id_level3_macd_dea_3','industry_id_level3_macd_3','industry_id_level3_macd_positive_ratio_3','industry_id_level3_macd_dif_5','industry_id_level3_macd_dea_5','industry_id_level3_macd_5','industry_id_level3_macd_positive_ratio_5','industry_id_level3_macd_dif_10','industry_id_level3_macd_dea_10','industry_id_level3_macd_10','industry_id_level3_macd_positive_ratio_10','industry_id_level3_macd_dif_20','industry_id_level3_macd_dea_20','industry_id_level3_macd_20','industry_id_level3_macd_positive_ratio_20','industry_id_level3_macd_dif_40','industry_id_level3_macd_dea_40','industry_id_level3_macd_40','industry_id_level3_macd_positive_ratio_40','industry_id_level3_macd_dif_dea','industry_id_level3_width_2','industry_id_level3_close_mb2_diff','industry_id_level3_width_3','industry_id_level3_close_mb3_diff','industry_id_level3_width_5','industry_id_level3_close_mb5_diff','industry_id_level3_width_10','industry_id_level3_close_mb10_diff','industry_id_level3_width_20','industry_id_level3_close_mb20_diff','industry_id_level3_width_40','industry_id_level3_close_mb40_diff','industry_id_level3_cr_bias_26d','industry_id_level3_cr_26d','industry_id_level3_cr_trend_26d','industry_id_level3_cr_trend_26d_0','industry_id_level3_cr_trend_26d_1','industry_id_level3_cr_trend_26d_2','industry_id_level3_cr_trend_26d_3','industry_id_level3_rsi_3d','industry_id_level3_rsi_5d','industry_id_level3_rsi_10d','industry_id_level3_rsi_20d','industry_id_level3_rsi_40d','industry_id_level3_turn_3d_avg','industry_id_level3_turn_3davg_dif','industry_id_level3_turn_3dmax_dif','industry_id_level3_turn_3dmin_dif','industry_id_level3_close_3davg_dif','industry_id_level3_close_3dmax_dif','industry_id_level3_close_3dmin_dif','industry_id_level3_close_3d_dif','industry_id_level3_turn_5d_avg','industry_id_level3_turn_5davg_dif','industry_id_level3_turn_5dmax_dif','industry_id_level3_turn_5dmin_dif','industry_id_level3_turn_3_5d_avg','industry_id_level3_turn_3_5dmax_dif','industry_id_level3_turn_3_5dmin_dif','industry_id_level3_close_5davg_dif','industry_id_level3_close_5dmax_dif','industry_id_level3_close_5dmin_dif','industry_id_level3_close_5d_dif','industry_id_level3_close_3_5d_avg','industry_id_level3_turn_10d_avg','industry_id_level3_turn_10davg_dif','industry_id_level3_turn_10dmax_dif','industry_id_level3_turn_10dmin_dif','industry_id_level3_turn_5_10d_avg','industry_id_level3_turn_5_10dmax_dif','industry_id_level3_turn_5_10dmin_dif','industry_id_level3_close_10davg_dif','industry_id_level3_close_10dmax_dif','industry_id_level3_close_10dmin_dif','industry_id_level3_close_10d_dif','industry_id_level3_close_5_10d_avg','industry_id_level3_turn_20d_avg','industry_id_level3_turn_20davg_dif','industry_id_level3_turn_20dmax_dif','industry_id_level3_turn_20dmin_dif','industry_id_level3_turn_10_20d_avg','industry_id_level3_turn_10_20dmax_dif','industry_id_level3_turn_10_20dmin_dif','industry_id_level3_close_20davg_dif','industry_id_level3_close_20dmax_dif','industry_id_level3_close_20dmin_dif','industry_id_level3_close_20d_dif','industry_id_level3_close_10_20d_avg','industry_id_level3_turn_30d_avg','industry_id_level3_turn_30davg_dif','industry_id_level3_turn_30dmax_dif','industry_id_level3_turn_30dmin_dif','industry_id_level3_turn_20_30d_avg','industry_id_level3_turn_20_30dmax_dif','industry_id_level3_turn_20_30dmin_dif','industry_id_level3_close_30davg_dif','industry_id_level3_close_30dmax_dif','industry_id_level3_close_30dmin_dif','industry_id_level3_close_30d_dif','industry_id_level3_close_20_30d_avg','industry_id_level3_turn_60d_avg','industry_id_level3_turn_60davg_dif','industry_id_level3_turn_60dmax_dif','industry_id_level3_turn_60dmin_dif','industry_id_level3_turn_30_60d_avg','industry_id_level3_turn_30_60dmax_dif','industry_id_level3_turn_30_60dmin_dif','industry_id_level3_close_60davg_dif','industry_id_level3_close_60dmax_dif','industry_id_level3_close_60dmin_dif','industry_id_level3_close_60d_dif','industry_id_level3_close_30_60d_avg','industry_id_level3_turn_120d_avg','industry_id_level3_turn_120davg_dif','industry_id_level3_turn_120dmax_dif','industry_id_level3_turn_120dmin_dif','industry_id_level3_turn_60_120d_avg','industry_id_level3_turn_60_120dmax_dif','industry_id_level3_turn_60_120dmin_dif','industry_id_level3_close_120davg_dif','industry_id_level3_close_120dmax_dif','industry_id_level3_close_120dmin_dif','industry_id_level3_close_120d_dif','industry_id_level3_close_60_120d_avg','industry_id_level3_turn_240d_avg','industry_id_level3_turn_240davg_dif','industry_id_level3_turn_240dmax_dif','industry_id_level3_turn_240dmin_dif','industry_id_level3_turn_120_240d_avg','industry_id_level3_turn_120_240dmax_dif','industry_id_level3_turn_120_240dmin_dif','industry_id_level3_close_240davg_dif','industry_id_level3_close_240dmax_dif','industry_id_level3_close_240dmin_dif','industry_id_level3_close_240d_dif','industry_id_level3_close_120_240d_avg','industry_id_level3_max_turn_index3d','industry_id_level3_max_close_index3d','industry_id_level3_min_close_index3d','industry_id_level3_max_turn_close3d','industry_id_level3_max_closs_turn3d','industry_id_level3_min_closs_turn3d','industry_id_level3_max_turn_index5d','industry_id_level3_max_close_index5d','industry_id_level3_min_close_index5d','industry_id_level3_max_turn_close5d','industry_id_level3_max_closs_turn5d','industry_id_level3_min_closs_turn5d','industry_id_level3_max_turn_index10d','industry_id_level3_max_close_index10d','industry_id_level3_min_close_index10d','industry_id_level3_max_turn_close10d','industry_id_level3_max_closs_turn10d','industry_id_level3_min_closs_turn10d','industry_id_level3_max_turn_index20d','industry_id_level3_max_close_index20d','industry_id_level3_min_close_index20d','industry_id_level3_max_turn_close20d','industry_id_level3_max_closs_turn20d','industry_id_level3_min_closs_turn20d','industry_id_level3_max_turn_index30d','industry_id_level3_max_close_index30d','industry_id_level3_min_close_index30d','industry_id_level3_max_turn_close30d','industry_id_level3_max_closs_turn30d','industry_id_level3_min_closs_turn30d','industry_id_level3_max_turn_index60d','industry_id_level3_max_close_index60d','industry_id_level3_min_close_index60d','industry_id_level3_max_turn_close60d','industry_id_level3_max_closs_turn60d','industry_id_level3_min_closs_turn60d','industry_id_level3_max_turn_index120d','industry_id_level3_max_close_index120d','industry_id_level3_min_close_index120d','industry_id_level3_max_turn_close120d','industry_id_level3_max_closs_turn120d','industry_id_level3_min_closs_turn120d','industry_id_level3_max_turn_index240d','industry_id_level3_max_close_index240d','industry_id_level3_min_close_index240d','industry_id_level3_max_turn_close240d','industry_id_level3_max_closs_turn240d','industry_id_level3_min_closs_turn240d']
      industry_id_level3_k_data = industry_id_level3_k_data[selected_columns]
      
      feature_all = pd.merge(feature_all, industry_id_level3_k_data, how="left", left_on=["date",'industry_id_level3'],right_on=["date",'industry_id_level3'])
      feature_all = feature_all.sort_values(['date', 'code'])
      del industry_id_level3_k_data
      gc.collect()
