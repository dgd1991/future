import copy
import gc
import os
import time
from itertools import zip_longest

import numpy
import pandas as pd

from feature.feature_process import *
from tools.Tools import Tools
from feature.get_technical_indicators import *
import numpy as np

pd.set_option('display.max_columns', 200)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)


class Feature(object):
   def __init__(self,  k_file_path, year, quarter_file_path, is_predict, date_start, date_end, model_name, is_code, is_industry1, is_industry2, is_industry3, is_market):
      self.k_file_path = k_file_path + str(year) + '.csv'
      self.k_file_path_his = k_file_path + str(year - 1) + '.csv'
      self.k_file_path_his2 = k_file_path + str(year - 2) + '.csv'
      self.quarter_file_path = quarter_file_path
      self.year = year
      self.tools = Tools()
      self.is_predict = is_predict
      self.date_start = date_start
      self.date_end = date_end
      self.model_name = model_name
      self.is_code = is_code
      self.is_industry1 = is_industry1
      self.is_industry2 = is_industry2
      self.is_industry3 = is_industry3
      self.is_market = is_market
      self.test = False
      self.code_feature_output_path = 'E:/pythonProject/future/data/datafile/feature/{model_name}/code_feature_{year}.csv'.format(model_name=model_name, year=str(year))
      self.industry1_feature_output_path = 'E:/pythonProject/future/data/datafile/feature/{model_name}/industry1_feature_{year}.csv'.format(model_name=model_name, year=str(year))
      self.industry2_feature_output_path = 'E:/pythonProject/future/data/datafile/feature/{model_name}/industry2_feature_{year}.csv'.format(model_name=model_name, year=str(year))
      self.industry3_feature_output_path = 'E:/pythonProject/future/data/datafile/feature/{model_name}/industry3_feature_{year}.csv'.format(model_name=model_name, year=str(year))
      self.market_feature_output_path = 'E:/pythonProject/future/data/datafile/feature/{model_name}/market_feature_{year}.csv'.format(model_name=model_name, year=str(year))

   def _rolling(self, _df, index_col, value_col, output_col, window_size):
      l1 = _df[index_col].tolist()
      l2 = _df[value_col].tolist()
      res = []
      lens = len(l1)
      for i in range(0, lens):
         index = i - (window_size - (l1[i] + 1))
         if index < 0 or np.isnan(index):
            res.append(None)
         else:
            res.append(l2[int(index)])
      _df[output_col] = res
      return _df

   def _object_rolling(self, _df, input_col, output_col, window_size, position):
      l1 = _df[input_col].tolist()
      res = []
      lens = len(l1)
      for i in range(0, lens):
         index = i - (window_size - position - 1)
         if index < 0 or np.isnan(index):
            res.append(None)
         else:
            res.append(l1[index])
      _df[output_col] = res
      return _df

   def func(self, _df, input_col, output_col):
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

   def feature_process(self):
      if self.is_code:
         raw_k_data = pd.read_csv(self.k_file_path)
         raw_k_data_his = pd.read_csv(self.k_file_path_his)
         raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
         del raw_k_data_his
         gc.collect()
         if year>2008:
            raw_k_data_his2 = pd.read_csv(self.k_file_path_his2)
            raw_k_data = pd.concat([raw_k_data_his2, raw_k_data], axis=0)
            del raw_k_data_his2
            gc.collect()
         if is_predict:
            raw_k_data = raw_k_data[raw_k_data['date']>self.tools.get_recent_month_date(self.date_start, -15)]
         if self.test:
            raw_k_data = raw_k_data.sample(frac=0.05)
         raw_k_data = raw_k_data[(raw_k_data['industry_id_level3'] > 0) | (raw_k_data['code'] == 'sh.000001') | (raw_k_data['code'] == 'sz.399001') | (raw_k_data['code'] == 'sz.399006')]
         raw_k_data["tradestatus"] = pd.to_numeric(raw_k_data["tradestatus"], errors='coerce')
         raw_k_data["turn"] = pd.to_numeric(raw_k_data["turn"], errors='coerce')
         raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')

         raw_k_data = raw_k_data[(raw_k_data['tradestatus'] == 1) & (raw_k_data['turn'] > 0)]
         raw_k_data = raw_k_data.sort_values(['date'])
         raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))
         raw_k_data['is_new'] = raw_k_data["pctChg"].groupby(level=0).apply(lambda x: x.rolling(min_periods=20, window=20, center=False).apply(lambda y: y[0]))
         raw_k_data = raw_k_data[raw_k_data['is_new'].map(lambda x: False if np.isnan(x) else True)]
         raw_k_data = raw_k_data.reset_index(level=0, drop=True)
         raw_k_data = raw_k_data.reset_index(level=0, drop=False)
         # raw_k_data = raw_k_data[raw_k_data['date']>=(str(year-2)+"-10-01")]
         
         raw_k_data["open"] = pd.to_numeric(raw_k_data["open"], errors='coerce')
         raw_k_data["close"] = pd.to_numeric(raw_k_data["close"], errors='coerce')
         raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')
         raw_k_data["preclose"] = pd.to_numeric(raw_k_data["preclose"], errors='coerce')
         raw_k_data["high"] = pd.to_numeric(raw_k_data["high"], errors='coerce')
         raw_k_data["low"] = pd.to_numeric(raw_k_data["low"], errors='coerce')
         raw_k_data['date_raw'] = raw_k_data['date']
         raw_k_data['date'] = pd.to_datetime(raw_k_data['date'])
         raw_k_data['open_ratio'] = ((raw_k_data['open'] - raw_k_data['preclose']) / raw_k_data['preclose'])
         raw_k_data['close_ratio'] = ((raw_k_data['close'] - raw_k_data['open']) / raw_k_data['open'])
         raw_k_data['high_ratio'] = ((raw_k_data['high'] - raw_k_data['preclose']) / raw_k_data['preclose'])
         raw_k_data['low_ratio'] = ((raw_k_data['low'] - raw_k_data['preclose']) / raw_k_data['preclose'])
         raw_k_data['pctChg'] = raw_k_data['pctChg'].map(lambda x: x/100.0)
         # 0.00000001 转化成亿
         raw_k_data['amount'] = pd.to_numeric(raw_k_data["amount"], errors='coerce')
         raw_k_data['market_value'] = 0.00000001 * raw_k_data['amount']/(raw_k_data['turn'].apply(lambda x: x/100))
         raw_k_data['code_market'] = raw_k_data['code'].map(lambda x: self.tools.code_market(x))

         raw_k_data['peTTM'] = raw_k_data['peTTM']
         raw_k_data['pcfNcfTTM'] = raw_k_data['pcfNcfTTM']
         raw_k_data['pbMRQ'] = raw_k_data['pbMRQ']
         raw_k_data['isST'] = raw_k_data['isST']

         raw_k_data = raw_k_data.sort_values(['date', 'code'])
         raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))
         raw_k_data.rename(columns={'date_raw': 'date'}, inplace=True)

         feature_all = copy.deepcopy(raw_k_data[['date', 'industry_name_level1','industry_name_level2','industry_name_level3','industry_id_level1','industry_id_level2','industry_id_level3','open_ratio','close_ratio','high_ratio','low_ratio','pctChg','code_market']])
         feature_all['open_ratio_7d_avg'] = raw_k_data.groupby(level=0)['open_ratio'].apply(lambda x: x.rolling(min_periods=7, window=7, center=False).mean())
         feature_all['close_ratio_7d_avg'] = raw_k_data.groupby(level=0)['close_ratio'].apply(lambda x: x.rolling(min_periods=7, window=7, center=False).mean())
         feature_all['high_ratio_7d_avg'] = raw_k_data.groupby(level=0)['high_ratio'].apply(lambda x: x.rolling(min_periods=7, window=7, center=False).mean())
         feature_all['low_ratio_7d_avg'] = raw_k_data.groupby(level=0)['low_ratio'].apply(lambda x: x.rolling(min_periods=7, window=7, center=False).mean())

         # feature_all['amount'] = raw_k_data['amount'].map(lambda x: None if x == '' else bignumber2Bucket(float(x) * 0.0000001, 1, 0))
         feature_all['market_value'] = raw_k_data['market_value'].map(lambda x: None if x == '' else bignumber2Bucket(float(x), 1.25, 60))
         # 一般市盈率10-30倍为好
         feature_all['peTTM'] = raw_k_data['peTTM'].map(lambda x: float2Bucket(float(x) + 200, 0.5, 0, 400, 200))
         # 市现率合适的范围是0-25之间
         feature_all['pcfNcfTTM'] = raw_k_data['pcfNcfTTM'].map(lambda x: float2Bucket(float(x) + 200, 0.5, 0, 400, 200))
         # 市净率一般在3 - 10之间是一个比较合理的范围
         feature_all['pbMRQ'] = raw_k_data['pbMRQ'].map(lambda x: float2Bucket(float(x), 2, 0, 100, 200))
         feature_all['isST'] = raw_k_data['isST'].map(lambda x: float2Bucket(float(x), 1, 0, 3, 3))
         feature_all['turn'] = raw_k_data['turn'].map(lambda x: float2Bucket(float(x), 2, 0, 50, 100))
         feature_all = feature_all.reset_index(level=0, drop=False)
         feature_all = feature_all.reset_index(level=0, drop=True)
         feature_all = feature_all[feature_all['date'] > str(self.year)]

         # kdj 5, 9, 19, 36, 45, 73，
         # 任意初始化，超过30天后的kdj值基本一样
         tmp = raw_k_data[['date', 'low', 'high', 'close']]
         tmp = tmp[tmp['date']>self.tools.get_recent_month_date(str(year)+"-01-01", -6)]
         kdj = tmp[['date']]
         for day_cnt in [3, 5, 9, 19, 73]:
            tmp['min'] = tmp['low'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).min())
            tmp['max'] = tmp['high'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).max())
            kdj['rsv_' + str(day_cnt)] = ((tmp['close'] - tmp['min'])/(tmp['max'] - tmp['min']))
            kdj['k_value_' + str(day_cnt)] = kdj['rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
            kdj['d_value_' + str(day_cnt)] = kdj['k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
            kdj['j_value_' + str(day_cnt)] = (3 * kdj['k_value_' + str(day_cnt)] - 2 * kdj['d_value_' + str(day_cnt)])
         kdj = kdj.reset_index(level=0, drop=False)
         kdj = kdj.reset_index(level=0, drop=True)
         kdj = kdj[kdj['date']>str(self.year)]
            # k_value_trend，kd_value效果不理想
            # feature_all['k_value_trend_' + str(day_cnt)] = feature_all['k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1])-y[0]))
            # feature_all['kd_value' + str(day_cnt)] = (feature_all['k_value_' + str(day_cnt)] - feature_all['d_value_' + str(day_cnt)]).rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1))

         # macd 12
         # 任意初始化，超过30天后的macd值基本一样
         tmp = raw_k_data[['date','close']]
         tmp = tmp[tmp['date'] > self.tools.get_recent_month_date(str(year) + "-01-01", -6)]
         macd = tmp[['date']]
         tmp['ema_12'] = tmp['close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 13, adjust=False).mean())
         tmp['ema_26'] = tmp['close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 27, adjust=False).mean())
         tmp['macd_dif'] = tmp['ema_12'] - tmp['ema_26']
         tmp['macd_dea'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0 / 5, adjust=False).mean())
         tmp['macd_dif_max'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=120, window=120, center=False).max())
         tmp['macd_dif_min'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=120, window=120, center=False).min())
         tmp['macd_dea_max'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=120, window=120, center=False).max())
         tmp['macd_dea_min'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=120, window=120, center=False).min())
         tmp['macd'] = (tmp['macd_dif'] - tmp['macd_dea']) * 2
         macd['macd_positive'] = tmp['macd'].apply(lambda x: 1 if x>0 else 0)
         macd['macd_dif_ratio'] = ((tmp['macd_dif']-tmp['macd_dif_min'])/(tmp['macd_dif_max']-tmp['macd_dif_min']))
         for day_cnt in [2, 3, 5, 10, 20, 40]:
            macd['macd_dif_' + str(day_cnt)] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
            macd['macd_dea_' + str(day_cnt)] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
            macd['macd_' + str(day_cnt)] = tmp['macd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
            macd['macd_positive_ratio_' + str(day_cnt)] = macd['macd_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
         macd['macd_dif_dea'] = (tmp['macd_dif']-tmp['macd_dea']).groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1)))
         macd = macd.reset_index(level=0, drop=False)
         macd = macd.reset_index(level=0, drop=True)
         macd = macd[macd['date'] > str(self.year)]

         # boll线
         # 效果很好，可以多做几天的，比如3天，5天，10天，40天
         tmp = raw_k_data[['date', 'close']]
         tmp = tmp[tmp['date'] > self.tools.get_recent_month_date(str(year) + "-01-01", -6)]
         boll = tmp[['date']]
         for day_cnt in [2, 3, 5, 10, 20, 40]:
            tmp['mb_' + str(day_cnt)] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            tmp['md_' + str(day_cnt)] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).std())
            tmp['up_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] + 2 * tmp['md_' + str(day_cnt)]
            tmp['dn_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] - 2 * tmp['md_' + str(day_cnt)]
            boll['width_' + str(day_cnt)] = (4 * tmp['md_' + str(day_cnt)] / tmp['mb_' + str(day_cnt)]).apply(lambda x: max(x, -3) if x<0 else min(x, 3))
            boll['close_mb' + str(day_cnt) + '_diff'] = ((tmp['close'] - tmp['mb_' + str(day_cnt)])/(2 * tmp['md_' + str(day_cnt)])).apply(lambda x: max(x, -3) if x<0 else min(x, 3))
         boll = boll.reset_index(level=0, drop=False)
         boll = boll.reset_index(level=0, drop=True)
         boll = boll[boll['date'] > str(self.year)]

         # cr指标
         # 似乎对中长期的指数cr指标效果较好，重新设计
         tmp = raw_k_data[['date', 'close', 'open', 'high', 'low']]
         tmp = tmp[tmp['date'] > self.tools.get_recent_month_date(str(year) + "-01-01", -4)]
         cr = tmp[['date']]
         tmp['cr_m'] = (tmp['close'] + tmp['open'] + tmp['high'] + tmp['low'])/4
         tmp['cr_ym'] = tmp['cr_m'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0]))
         tmp['cr_p1_day'] = (tmp['high'] - tmp['cr_ym']).apply(lambda x: max(x, 0))
         tmp['cr_p2_day'] = (tmp['cr_ym'] - tmp['low']).apply(lambda x: max(x, 0))
         for day_cnt in [26]:
            tmp['cr_p1_' + str(day_cnt) + 'd'] = tmp['cr_p1_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).sum())
            tmp['cr_p2_' + str(day_cnt) + 'd'] = tmp['cr_p2_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).sum())
            tmp['cr_' + str(day_cnt) + 'd'] = tmp['cr_p1_' + str(day_cnt) + 'd'] / tmp['cr_p2_' + str(day_cnt) + 'd']
            tmp['cr_a_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=5, window=5, center=False).mean())
            tmp['cr_b_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=10, window=10, center=False).mean())
            tmp['cr_c_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=20, window=20, center=False).mean())
            tmp['cr_d_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=60, window=60, center=False).mean())
            cr['cr_bias_' + str(day_cnt) + 'd'] = (tmp['cr_' + str(day_cnt) + 'd']/tmp['cr_a_' + str(day_cnt) + 'd']).apply(lambda x: max(x, -3) if x<0 else min(x, 3))
            cr['cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: float2Bucket(float(x)*100, 0.1, 0, 300, 30))

            tmp['cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
            tmp['cr_a_' + str(day_cnt) + 'd'] = tmp['cr_a_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
            tmp['cr_b_' + str(day_cnt) + 'd'] = tmp['cr_b_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
            tmp['cr_c_' + str(day_cnt) + 'd'] = tmp['cr_c_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
            tmp['cr_d_' + str(day_cnt) + 'd'] = tmp['cr_d_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
            # bucket空间可以设置成 75万，多个特征可以共享embedding
            cr['cr_trend_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].str.cat([tmp['cr_a_' + str(day_cnt) + 'd'],tmp['cr_b_' + str(day_cnt) + 'd'],tmp['cr_c_' + str(day_cnt) + 'd'],tmp['cr_d_' + str(day_cnt) + 'd']], sep='_').apply(lambda x: self.tools.hash_bucket(x, 750000))
            for day_cnt_new in range(4):
               cr['cr_trend_' + str(day_cnt) + 'd' + '_' + str(day_cnt_new)] = cr['cr_trend_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt_new+2, window=day_cnt_new+2, center=False).apply(lambda y: y[0]))
         cr = cr.reset_index(level=0, drop=False)
         cr = cr.reset_index(level=0, drop=True)
         cr = cr[cr['date'] > str(self.year)]

         # rsi指标
         tmp = raw_k_data[['date', 'close', 'preclose']]
         tmp = tmp[tmp['date'] > self.tools.get_recent_month_date(str(year) + "-01-01", -6)]
         rsi = tmp[['date']]
         tmp['price_dif'] = tmp['close'] - tmp['preclose']
         tmp['rsi_positive'] = tmp['price_dif'].apply(lambda x: max(x, 0))
         tmp['rsi_all'] = tmp['price_dif'].apply(lambda x: abs(x))
         for day_cnt in (3, 5, 10, 20, 40):
            rsi['rsi_' + str(day_cnt) + 'd'] = (tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).sum()) / tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).sum()))
         rsi = rsi.reset_index(level=0, drop=False)
         rsi = rsi.reset_index(level=0, drop=True)
         rsi = rsi[rsi['date'] > str(self.year)]

         # 成交量和换手率特征
         tmp = raw_k_data[['date', 'close', 'preclose','turn']]
         tmp = tmp[tmp['date'] > self.tools.get_recent_month_date(str(year) + "-01-01", -15)]
         turn_close = tmp[['date']]
         day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
         for index in range(len(day_cnt_list)):
            day_cnt = day_cnt_list[index]
            day_cnt_last = day_cnt_list[index-1]
            turn_close['turn_' + str(day_cnt) + 'd' + '_avg'] = tmp.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            # turn_close['turn_rank_' + str(day_cnt) + 'd'] = turn_close['turn_' + str(day_cnt) + 'd' + '_avg']
            turn_close['turn_' + str(day_cnt) + 'd' + 'avg_dif'] = tmp['turn']/turn_close['turn_' + str(day_cnt) + 'd' + '_avg']
            turn_close['turn_' + str(day_cnt) + 'd' + 'max_dif'] = turn_close['turn_' + str(day_cnt) + 'd' + '_avg']/tmp.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).max())
            turn_close['turn_' + str(day_cnt) + 'd' + 'min_dif'] = tmp.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).min())/turn_close['turn_' + str(day_cnt) + 'd' + '_avg']
            if index>0:
               turn_close['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = turn_close['turn_' + str(day_cnt_last) + 'd' + '_avg']/turn_close['turn_' + str(day_cnt) + 'd' + '_avg']
               turn_close['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'] = turn_close['turn_' + str(day_cnt_last) + 'd' + 'max_dif']/turn_close['turn_' + str(day_cnt) + 'd' + 'max_dif']
               turn_close['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'] = turn_close['turn_' + str(day_cnt_last) + 'd' + 'min_dif']/turn_close['turn_' + str(day_cnt) + 'd' + 'min_dif']

            tmp['close_' + str(day_cnt) + 'd' + '_avg'] = tmp.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            turn_close['close_' + str(day_cnt) + 'd' + 'avg_dif'] = tmp['close']/tmp['close_' + str(day_cnt) + 'd' + '_avg']
            turn_close['close_' + str(day_cnt) + 'd' + 'max_dif'] = tmp['close']/tmp.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).max())
            turn_close['close_' + str(day_cnt) + 'd' + 'min_dif'] = tmp['close']/tmp.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).min())
            turn_close['close_' + str(day_cnt) + 'd' + '_dif'] = tmp.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 1.0*(y[day_cnt-1])/y[0]))
            if index>0:
               turn_close['close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = tmp['close_' + str(day_cnt_last) + 'd' + '_avg']/tmp['close_' + str(day_cnt) + 'd' + '_avg']

         for index in range(len(day_cnt_list)):
            day_cnt = day_cnt_list[index]
            day_cnt_last = day_cnt_list[index - 1]
            if index > 0:
               turn_close['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = turn_close['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 20, 0, 10, 200))
               turn_close['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'] = turn_close['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 20, 0, 10, 200))
               turn_close['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'] = turn_close['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 20, 0, 10, 200))
               turn_close['close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = turn_close['close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 4, 200))
            turn_close['turn_' + str(day_cnt) + 'd' + '_avg'] = turn_close['turn_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: min(2, round(x/100, 4)))
            turn_close['turn_' + str(day_cnt) + 'd' + 'avg_dif'] = turn_close['turn_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
            turn_close['turn_' + str(day_cnt) + 'd' + 'max_dif'] = turn_close['turn_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: round(x, 4))
            turn_close['turn_' + str(day_cnt) + 'd' + 'min_dif'] = turn_close['turn_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: round(x, 4))

            turn_close['close_' + str(day_cnt) + 'd' + 'avg_dif'] = turn_close['close_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
            turn_close['close_' + str(day_cnt) + 'd' + 'max_dif'] = turn_close['close_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 100, 0, 1, 100))
            turn_close['close_' + str(day_cnt) + 'd' + 'min_dif'] = turn_close['close_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 1, 10, 500))
            turn_close['close_' + str(day_cnt) + 'd' + '_dif'] = turn_close['close_' + str(day_cnt) + 'd' + '_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

         # 还需要再做一些量价类特征，比如高位放量，低位缩量等等
         for day_cnt in [3, 5, 10, 20, 30, 60, 120, 240]:
            tmp['max_turn_index' + str(day_cnt) + 'd'] = tmp['turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: y.argmax()))
            tmp['max_close_index' + str(day_cnt) + 'd'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: y.argmax()))
            tmp['min_close_index' + str(day_cnt) + 'd'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: y.argmin()))
            tmp = tmp.groupby(level=0).apply(lambda x: self._rolling(x, 'max_turn_index' + str(day_cnt) + 'd', 'close', 'max_turn_close' + str(day_cnt) + 'd', day_cnt))
            tmp = tmp.groupby(level=0).apply(lambda x: self._rolling(x, 'max_close_index' + str(day_cnt) + 'd', 'turn', 'max_closs_turn' + str(day_cnt) + 'd', day_cnt))
            tmp = tmp.groupby(level=0).apply(lambda x: self._rolling(x, 'min_close_index' + str(day_cnt) + 'd', 'turn', 'min_closs_turn' + str(day_cnt) + 'd', day_cnt))

            turn_close['max_turn_index' + str(day_cnt) + 'd'] = tmp['max_turn_index' + str(day_cnt) + 'd']
            turn_close['max_close_index' + str(day_cnt) + 'd'] = tmp['max_close_index' + str(day_cnt) + 'd']
            turn_close['min_close_index' + str(day_cnt) + 'd'] = tmp['max_close_index' + str(day_cnt) + 'd']
            turn_close['max_turn_close' + str(day_cnt) + 'd'] = (tmp['close']/tmp['max_turn_close' + str(day_cnt) + 'd']).map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
            turn_close['max_closs_turn' + str(day_cnt) + 'd'] = (tmp['max_closs_turn' + str(day_cnt) + 'd']/tmp['turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
            turn_close['min_closs_turn' + str(day_cnt) + 'd'] = (tmp['min_closs_turn' + str(day_cnt) + 'd']/tmp['turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
         turn_close = turn_close.reset_index(level=0, drop=False)
         turn_close = turn_close.reset_index(level=0, drop=True)
         turn_close = turn_close[turn_close['date'] > str(self.year)]

         # 新增过去n天每个涨幅段的天数比例，换手率的比例,涨停的天数,主要捕捉股性特征，判断股票是否活跃
         tmp = raw_k_data[['date','pctChg', 'turn', 'close', 'high', 'low', 'isST']]
         tmp = tmp[tmp['date'] > self.tools.get_recent_month_date(str(year) + "-01-01", -10)]
         code_activity = tmp[['date']]
         tmp['pctChg_limit'] = tmp.apply(lambda x: self.tools.code_pctChg_limit_type(x.pctChg, x.isST, x.high, x.low, x.close), axis=1)
         tmp['pctChg_up_limit'] = tmp['pctChg_limit'].map(lambda x: 1 if x == 1 else 0)
         tmp['pctChg_down_limit'] = tmp['pctChg_limit'].map(lambda x: 1 if x == 2 else 0)
         tmp['pctChg_greater_3'] = tmp['pctChg'].map(lambda x: 1 if x > 0.03 else 0)
         tmp['pctChg_greater_6'] = tmp['pctChg'].map(lambda x: 1 if x > 0.06 else 0)
         tmp['pctChg_greater_9'] = tmp['pctChg'].map(lambda x: 1 if x > 0.09 else 0)
         tmp['pctChg_greater_13'] = tmp['pctChg'].map(lambda x: 1 if x > 0.13 else 0)

         tmp['pctChg_less_3'] = tmp['pctChg'].map(lambda x: 1 if x < -0.03 else 0)
         tmp['pctChg_less_6'] = tmp['pctChg'].map(lambda x: 1 if x < -0.06 else 0)
         tmp['pctChg_less_9'] = tmp['pctChg'].map(lambda x: 1 if x < -0.09 else 0)
         tmp['pctChg_less_13'] = tmp['pctChg'].map(lambda x: 1 if x < -0.13 else 0)

         tmp['turn_greater_3'] = tmp['turn'].map(lambda x: 1 if x > 3 else 0)
         tmp['turn_greater_6'] = tmp['turn'].map(lambda x: 1 if x > 6 else 0)
         tmp['turn_greater_10'] = tmp['turn'].map(lambda x: 1 if x > 10 else 0)
         tmp['turn_greater_15'] = tmp['turn'].map(lambda x: 1 if x > 15 else 0)
         tmp['turn_greater_21'] = tmp['turn'].map(lambda x: 1 if x > 21 else 0)

         for day_cnt in [3, 5, 10, 20, 30, 60, 120]:
            code_activity['pctChg_up_limit_' + str(day_cnt) + 'd'] = tmp['pctChg_up_limit'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            code_activity['pctChg_down_limit_' + str(day_cnt) + 'd'] = tmp['pctChg_down_limit'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())

            code_activity['pctChg_greater_3_' + str(day_cnt) + 'd'] = tmp['pctChg_greater_3'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            code_activity['pctChg_greater_6_' + str(day_cnt) + 'd'] = tmp['pctChg_greater_6'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            code_activity['pctChg_greater_9_' + str(day_cnt) + 'd'] = tmp['pctChg_greater_9'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            code_activity['pctChg_greater_13_' + str(day_cnt) + 'd'] = tmp['pctChg_greater_13'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())

            code_activity['pctChg_less_3_' + str(day_cnt) + 'd'] = tmp['pctChg_less_3'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            code_activity['pctChg_less_6_' + str(day_cnt) + 'd'] = tmp['pctChg_less_6'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            code_activity['pctChg_less_9_' + str(day_cnt) + 'd'] = tmp['pctChg_less_9'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            code_activity['pctChg_less_13_' + str(day_cnt) + 'd'] = tmp['pctChg_less_13'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())

            code_activity['turn_greater_3_' + str(day_cnt) + 'd'] = tmp['turn_greater_3'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            code_activity['turn_greater_6_' + str(day_cnt) + 'd'] = tmp['turn_greater_6'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            code_activity['turn_greater_10_' + str(day_cnt) + 'd'] = tmp['turn_greater_10'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            code_activity['turn_greater_15_' + str(day_cnt) + 'd'] = tmp['turn_greater_15'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            code_activity['turn_greater_21_' + str(day_cnt) + 'd'] = tmp['turn_greater_21'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
         code_activity = code_activity.reset_index(level=0, drop=False)
         code_activity = code_activity.reset_index(level=0, drop=True)
         code_activity = code_activity[code_activity['date'] > str(self.year)]

         # 新增过去n天股票的涨幅，以及在全市场的排名
         del tmp
         gc.collect()
         tmp = raw_k_data[['date',"industry_id_level1","industry_id_level2","industry_id_level3",'turn', 'close']]
         tmp = tmp[tmp['date'] > self.tools.get_recent_month_date(str(year) + "-01-01", -10)]
         code_pctChg_nd = tmp[['date']]
         for day_cnt in [3, 5, 10, 20, 30, 60, 120]:
            code_pctChg_nd['pctChg_' + str(day_cnt) + 'd'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: (y[day_cnt-1]-y[0])/y[0]))
            tmp['pctChg_' + str(day_cnt) + 'd'] = code_pctChg_nd['pctChg_' + str(day_cnt) + 'd']
            # 新增过去n天涨幅超过大盘，行业指数m个点的天数
         # 新增板块的最近n天的涨停股票数量，主要评估板块热度
         code_pctChg_nd = code_pctChg_nd.reset_index(level=0, drop=False)
         code_pctChg_nd = code_pctChg_nd.reset_index(level=0, drop=True)
         code_pctChg_nd = code_pctChg_nd[code_pctChg_nd['date'] > str(self.year)]

         tmp = tmp.reset_index(level=0, drop=False)
         tmp = tmp.reset_index(level=0, drop=True)
         code_activity_rank = tmp[['date','code']]
         for day_cnt in [3, 5, 10, 20, 30, 60, 120]:
            tmp['pctChg_' + str(day_cnt) + 'd'] = tmp['pctChg_' + str(day_cnt) + 'd']
            tmp['pctChg_rank' + str(day_cnt) + 'd'] = tmp['pctChg_' + str(day_cnt) + 'd'].groupby(tmp['date']).rank(ascending=False)

            tmp['pctChg_rank_industry1_' + str(day_cnt) + 'd'] = tmp.sort_values(['pctChg_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level1']).cumcount()+1
            tmp['pctChg_rank_industry2_' + str(day_cnt) + 'd'] = tmp.sort_values(['pctChg_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level2']).cumcount()+1
            tmp['pctChg_rank_industry3_' + str(day_cnt) + 'd'] = tmp.sort_values(['pctChg_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level3']).cumcount()+1

            if day_cnt == 3:
               tmp['code_count'] = tmp.groupby('date')['pctChg_rank' + str(day_cnt) + 'd'].transform('max')
               tmp['industry1_count'] = tmp.groupby(['date','industry_id_level1'])['pctChg_rank_industry1_' + str(day_cnt) + 'd'].transform('max')
               tmp['industry2_count'] = tmp.groupby(['date','industry_id_level2'])['pctChg_rank_industry2_' + str(day_cnt) + 'd'].transform('max')
               tmp['industry3_count'] = tmp.groupby(['date','industry_id_level3'])['pctChg_rank_industry3_' + str(day_cnt) + 'd'].transform('max')

            code_activity_rank['pctChg_rank_ratio' + str(day_cnt) + 'd'] = (tmp['pctChg_rank' + str(day_cnt) + 'd']/tmp['code_count'])
            code_activity_rank['pctChg_rank_ratio_industry1_' + str(day_cnt) + 'd'] = (tmp['pctChg_rank_industry1_' + str(day_cnt) + 'd']/tmp['industry1_count'])
            code_activity_rank['pctChg_rank_ratio_industry2_' + str(day_cnt) + 'd'] = (tmp['pctChg_rank_industry2_' + str(day_cnt) + 'd']/tmp['industry2_count'])
            code_activity_rank['pctChg_rank_ratio_industry3_' + str(day_cnt) + 'd'] = (tmp['pctChg_rank_industry3_' + str(day_cnt) + 'd']/tmp['industry3_count'])

            tmp['turn_rank_' + str(day_cnt) + 'd'] = tmp[['code', 'turn']].groupby(['code'])['turn'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            tmp['turn_rank_' + str(day_cnt) + 'd'] = tmp['turn_rank_' + str(day_cnt) + 'd'].groupby(tmp['date']).rank(ascending=False)

            tmp['turn_rank_industry1_' + str(day_cnt) + 'd'] = tmp.sort_values(['turn_rank_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level1']).cumcount()+1
            tmp['turn_rank_industry2_' + str(day_cnt) + 'd'] = tmp.sort_values(['turn_rank_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level2']).cumcount()+1
            tmp['turn_rank_industry3_' + str(day_cnt) + 'd'] = tmp.sort_values(['turn_rank_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level3']).cumcount()+1

            code_activity_rank['turn_rank_' + str(day_cnt) + 'd'] = (tmp['turn_rank_' + str(day_cnt) + 'd']/tmp['code_count'])
            code_activity_rank['turn_rank_industry1_' + str(day_cnt) + 'd'] = (tmp['turn_rank_industry1_' + str(day_cnt) + 'd']/tmp['code_count'])
            code_activity_rank['turn_rank_industry2_' + str(day_cnt) + 'd'] = (tmp['turn_rank_industry2_' + str(day_cnt) + 'd']/tmp['code_count'])
            code_activity_rank['turn_rank_industry3_' + str(day_cnt) + 'd'] = (tmp['turn_rank_industry3_' + str(day_cnt) + 'd']/tmp['code_count'])
         code_activity_rank = code_activity_rank[code_activity_rank['date'] > str(self.year)]

         # 公司市值排名特征
         del tmp
         gc.collect()
         tmp = raw_k_data[['date','industry_id_level1','industry_id_level2','industry_id_level3', 'market_value']]
         tmp = tmp.reset_index(level=0, drop=False)
         tmp = tmp.reset_index(level=0, drop=True)
         tmp = tmp[tmp['date'] > str(self.year)]
         tmp = tmp.sort_values(['date', 'code'])

         tmp['market_value_rank_industry1'] = tmp.sort_values(['market_value'], ascending=False).groupby(['date','industry_id_level1']).cumcount()+1
         tmp['market_value_rank_industry2'] = tmp.sort_values(['market_value'], ascending=False).groupby(['date','industry_id_level2']).cumcount()+1
         tmp['market_value_rank_industry3'] = tmp.sort_values(['market_value'], ascending=False).groupby(['date','industry_id_level3']).cumcount()+1
         tmp['industry1_count'] = tmp.groupby(['date','industry_id_level1'])['market_value_rank_industry1'].transform('max')
         tmp['industry2_count'] = tmp.groupby(['date','industry_id_level2'])['market_value_rank_industry2'].transform('max')
         tmp['industry3_count'] = tmp.groupby(['date','industry_id_level3'])['market_value_rank_industry3'].transform('max')
         tmp['market_value_rank_ratio_industry1'] = (tmp['market_value_rank_industry1']/tmp['industry1_count'])
         tmp['market_value_rank_ratio_industry2'] = (tmp['market_value_rank_industry2']/tmp['industry2_count'])
         tmp['market_value_rank_ratio_industry3'] = (tmp['market_value_rank_industry3']/tmp['industry3_count'])
         market_value_rank_fea = tmp[['code', 'date', 'market_value_rank_ratio_industry1', 'market_value_rank_ratio_industry2', 'market_value_rank_ratio_industry3']]
         del tmp
         gc.collect()

         # 过去n天涨幅/成交额、换手率排名前10%的天数比例
         tmp = raw_k_data[['date', 'pctChg', 'amount', 'turn']]
         tmp = tmp.reset_index(level=0, drop=False)
         tmp = tmp.reset_index(level=0, drop=True)
         tmp = tmp[tmp['date'] > self.tools.get_recent_month_date(str(year) + "-01-01", -15)]
         tmp = tmp.sort_values(['date', 'code'])

         tmp['pctChg_rank'] = tmp.sort_values(['pctChg'], ascending=False).groupby(['date']).cumcount()+1
         tmp['amount_rank'] = tmp.sort_values(['amount'], ascending=False).groupby(['date']).cumcount() + 1
         tmp['turn_rank'] = tmp.sort_values(['turn'], ascending=False).groupby(['date']).cumcount() + 1
         tmp['code_count'] = tmp.groupby('date')['pctChg_rank'].transform('max')
         tmp['pctChg_rank_ratio'] = tmp['pctChg_rank']/tmp['code_count']
         tmp['amount_rank_ratio'] = tmp['amount_rank'] / tmp['code_count']
         tmp['turn_rank_ratio'] = tmp['turn_rank'] / tmp['code_count']
         tmp['pctChg_top1'] = tmp['pctChg_rank_ratio'].map(lambda x: 1 if x<0.01 else 0)
         tmp['pctChg_top5'] = tmp['pctChg_rank_ratio'].map(lambda x: 1 if x < 0.05 else 0)
         tmp['pctChg_top10'] = tmp['pctChg_rank_ratio'].map(lambda x: 1 if x < 0.1 else 0)
         tmp['pctChg_top20'] = tmp['pctChg_rank_ratio'].map(lambda x: 1 if x < 0.2 else 0)
         tmp['pctChg_top30'] = tmp['pctChg_rank_ratio'].map(lambda x: 1 if x < 0.3 else 0)

         tmp['amount_top1'] = tmp['amount_rank_ratio'].map(lambda x: 1 if x<0.01 else 0)
         tmp['amount_top5'] = tmp['amount_rank_ratio'].map(lambda x: 1 if x < 0.05 else 0)
         tmp['amount_top10'] = tmp['amount_rank_ratio'].map(lambda x: 1 if x < 0.1 else 0)

         tmp['turn_top1'] = tmp['turn_rank_ratio'].map(lambda x: 1 if x<0.01 else 0)
         tmp['turn_top5'] = tmp['turn_rank_ratio'].map(lambda x: 1 if x < 0.05 else 0)
         tmp['turn_top10'] = tmp['turn_rank_ratio'].map(lambda x: 1 if x < 0.1 else 0)
         tmp['turn_top20'] = tmp['turn_rank_ratio'].map(lambda x: 1 if x < 0.2 else 0)
         tmp['turn_top30'] = tmp['turn_rank_ratio'].map(lambda x: 1 if x < 0.3 else 0)
         topN_percent_ratio = tmp[['date', 'code']]
         for day_cnt in [1, 3, 5, 8, 15, 30, 60, 120, 240]:
            topN_percent_ratio['pctChg_top1_' + str(day_cnt) + 'd'] = tmp[['pctChg_top1','code']].groupby(['code'])['pctChg_top1'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            topN_percent_ratio['pctChg_top5_' + str(day_cnt) + 'd'] = tmp[['pctChg_top5','code']].groupby(['code'])['pctChg_top5'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            topN_percent_ratio['pctChg_top10_' + str(day_cnt) + 'd'] = tmp[['pctChg_top10','code']].groupby(['code'])['pctChg_top10'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            topN_percent_ratio['pctChg_top20_' + str(day_cnt) + 'd'] = tmp[['pctChg_top20','code']].groupby(['code'])['pctChg_top20'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            topN_percent_ratio['pctChg_top30_' + str(day_cnt) + 'd'] = tmp[['pctChg_top30','code']].groupby(['code'])['pctChg_top30'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())

            topN_percent_ratio['amount_top1_' + str(day_cnt) + 'd'] = tmp[['amount_top1','code']].groupby(['code'])['amount_top1'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            topN_percent_ratio['amount_top5_' + str(day_cnt) + 'd'] = tmp[['amount_top5','code']].groupby(['code'])['amount_top5'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            topN_percent_ratio['amount_top10_' + str(day_cnt) + 'd'] = tmp[['amount_top10','code']].groupby(['code'])['amount_top10'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())

            topN_percent_ratio['turn_top1_' + str(day_cnt) + 'd'] = tmp[['turn_top1','code']].groupby(['code'])['turn_top1'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            topN_percent_ratio['turn_top5_' + str(day_cnt) + 'd'] = tmp[['turn_top5','code']].groupby(['code'])['turn_top5'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            topN_percent_ratio['turn_top10_' + str(day_cnt) + 'd'] = tmp[['turn_top10','code']].groupby(['code'])['turn_top10'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            topN_percent_ratio['turn_top20_' + str(day_cnt) + 'd'] = tmp[['turn_top20','code']].groupby(['code'])['turn_top20'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            topN_percent_ratio['turn_top30_' + str(day_cnt) + 'd'] = tmp[['turn_top30','code']].groupby(['code'])['turn_top30'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())

         topN_percent_ratio = topN_percent_ratio[topN_percent_ratio['date'] > str(self.year)]
         del tmp
         gc.collect()

         # 合并特征
         del raw_k_data
         gc.collect()
         feature_all = pd.merge(feature_all, kdj, how="left", left_on=['date', 'code'], right_on=['date', 'code'])
         del kdj
         gc.collect()
         feature_all = pd.merge(feature_all, macd, how="left", left_on=['date', 'code'], right_on=['date', 'code'])
         del macd
         gc.collect()
         feature_all = pd.merge(feature_all, boll, how="left", left_on=['date', 'code'], right_on=['date', 'code'])
         del boll
         gc.collect()
         feature_all = pd.merge(feature_all, cr, how="left", left_on=['date', 'code'], right_on=['date', 'code'])
         del cr
         gc.collect()
         feature_all = pd.merge(feature_all, rsi, how="left", left_on=['date', 'code'], right_on=['date', 'code'])
         del rsi
         gc.collect()
         feature_all = pd.merge(feature_all, turn_close, how="left", left_on=['date', 'code'], right_on=['date', 'code'])
         del turn_close
         gc.collect()
         feature_all = pd.merge(feature_all, code_activity, how="left", left_on=['date', 'code'], right_on=['date', 'code'])
         del code_activity
         gc.collect()
         feature_all = pd.merge(feature_all, code_pctChg_nd, how="left", left_on=['date', 'code'], right_on=['date', 'code'])
         del code_pctChg_nd
         gc.collect()
         feature_all = pd.merge(feature_all, code_activity_rank, how="left", left_on=['date', 'code'], right_on=['date', 'code'])
         del code_activity_rank
         gc.collect()
         feature_all = pd.merge(feature_all, market_value_rank_fea, how="left", left_on=['date', 'code'], right_on=['date', 'code'])
         del market_value_rank_fea
         gc.collect()
         feature_all = pd.merge(feature_all, topN_percent_ratio, how="left", left_on=['date', 'code'], right_on=['date', 'code'])
         del topN_percent_ratio
         gc.collect()

         if is_predict:
            feature_all = feature_all[(feature_all['date'] >= self.date_start) & (feature_all['date'] <= self.date_end)]
         feature_all = feature_all[feature_all['date'] > str(self.year)]
         feature_all = feature_all.round(5)
         if os.path.isfile(self.code_feature_output_path):
            feature_all.to_csv(self.code_feature_output_path, mode='a', header=False, index=False)
         else:
            feature_all.to_csv(self.code_feature_output_path, mode='w', header=True, index=False)

      if self.is_predict:
         sample = feature_all
         sample['date'] = sample['date'].map(lambda x: int(str(x).replace('-', '')[:6]))
         sample = sample[sample['code_market'] != 0]
         sample['label_7'] = 0
         sample['pctChg_7'] = 0
         sample['sh_pctChg_7'] = 0
         sample['label_15'] = 0
         sample['pctChg_15'] = 0
         sample['sh_pctChg_15'] = 0
         sample.to_csv('E:/pythonProject/future/data/datafile/prediction_sample/{model_name}/prediction_sample_{date}.csv'.format(model_name=self.model_name, date=str(self.date_end)), mode='a',header=True, index=False, encoding='utf-8')

if __name__ == '__main__':
   is_code = True
   # is_industry1 = True
   # is_industry2 = True
   # is_industry3 = True
   # is_market = True
   
   # is_code = False
   is_industry1 = False
   is_industry2 = False
   is_industry3 = False
   is_market = False

   years = [2008]
   is_predict = False
   model_name = 'model_v13'
   date_start = '2023-04-17'
   date_end = '2023-04-17'
   years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
   # time.sleep(18000)
   # years = [2020,2021,2022]

   for year in years:
      path = 'E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_'
      quater_path = 'E:/pythonProject/future/data/datafile/code_quarter_data_v2_all.csv'
      output_path = 'E:/pythonProject/future/data/datafile/feature/{model_name}/{year}_feature.csv'.format(model_name=model_name, year=str(year))
      # raw_k_data = pd.read_csv(path + str(year) + '.csv')
      # raw_k_data.to_csv('E:/pythonProject/future/data/datafile/raw_feature/test_code_k_data_v5_' + str(year) + '.csv', mode='a', header=True, index=False)
      feature = Feature(path, year, quater_path, is_predict, date_start, date_end, model_name, is_code, is_industry1, is_industry2, is_industry3, is_market)
      feature.feature_process()
      # feature_all = feature_all
      # if os.path.isfile(output_path):
      #    feature_all.to_csv(output_path, mode='a', header=False, index=False)
      # else:
      #    feature_all.to_csv(output_path, mode='w', header=True, index=False)
      # del feature_all
      # gc.collect()