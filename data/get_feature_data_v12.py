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
         raw_k_data = raw_k_data[raw_k_data['date']>=(str(year-2)+"-10-01")]
         
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
         raw_k_data['pctChg'] = raw_k_data['pctChg'].map(lambda x: x/100.0)
         # 0.00000001 转化成亿
         raw_k_data['amount'] = pd.to_numeric(raw_k_data["amount"], errors='coerce')
         raw_k_data['market_value'] = 0.00000001 * raw_k_data['amount']/(raw_k_data['turn'].apply(lambda x: x/100))
         raw_k_data['code_market'] = raw_k_data['code'].map(lambda x: self.tools.code_market(x))

         raw_k_data['peTTM'] = raw_k_data['peTTM']
         raw_k_data['pcfNcfTTM'] = raw_k_data['pcfNcfTTM']
         raw_k_data['pbMRQ'] = raw_k_data['pbMRQ']
         raw_k_data['isST'] = raw_k_data['isST']

         raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))

         feature_all = copy.deepcopy(raw_k_data[['industry_name_level1','industry_name_level2','industry_name_level3','industry_id_level1','industry_id_level2','industry_id_level3','open_ratio','close_ratio','high_ratio','low_ratio','pctChg','code_market']])
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

         # kdj 5, 9, 19, 36, 45, 73，
         # 任意初始化，超过30天后的kdj值基本一样
         tmp = raw_k_data[['low', 'high', 'close']]
         for day_cnt in [3, 5, 9, 19, 73]:
            tmp['min'] = tmp['low'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).min())
            tmp['max'] = tmp['high'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).max())
            feature_all['rsv_' + str(day_cnt)] = ((tmp['close'] - tmp['min'])/(tmp['max'] - tmp['min']))
            feature_all['k_value_' + str(day_cnt)] = feature_all['rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
            feature_all['d_value_' + str(day_cnt)] = feature_all['k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean())
            feature_all['j_value_' + str(day_cnt)] = (3 * feature_all['k_value_' + str(day_cnt)] - 2 * feature_all['d_value_' + str(day_cnt)])
            # k_value_trend，kd_value效果不理想
            # feature_all['k_value_trend_' + str(day_cnt)] = feature_all['k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1])-y[0]))
            # feature_all['kd_value' + str(day_cnt)] = (feature_all['k_value_' + str(day_cnt)] - feature_all['d_value_' + str(day_cnt)]).rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1))

         # macd 12
         # 任意初始化，超过30天后的macd值基本一样
         tmp = raw_k_data[['close']]
         tmp['ema_12'] = tmp['close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 13, adjust=False).mean())
         tmp['ema_26'] = tmp['close'].groupby(level=0).apply(lambda x: x.ewm(alpha=2.0 / 27, adjust=False).mean())
         tmp['macd_dif'] = tmp['ema_12'] - tmp['ema_26']
         tmp['macd_dea'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0 / 5, adjust=False).mean())
         tmp['macd_dif_max'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=120, window=120, center=False).max())
         tmp['macd_dif_min'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=120, window=120, center=False).min())
         tmp['macd_dea_max'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=120, window=120, center=False).max())
         tmp['macd_dea_min'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=120, window=120, center=False).min())
         tmp['macd'] = (tmp['macd_dif'] - tmp['macd_dea']) * 2
         feature_all['macd_positive'] = tmp['macd'].apply(lambda x: 1 if x>0 else 0)
         feature_all['macd_dif_ratio'] = ((tmp['macd_dif']-tmp['macd_dif_min'])/(tmp['macd_dif_max']-tmp['macd_dif_min']))
         for day_cnt in [2, 3, 5, 10, 20, 40]:
            feature_all['macd_dif_' + str(day_cnt)] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
            feature_all['macd_dea_' + str(day_cnt)] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
            feature_all['macd_' + str(day_cnt)] = tmp['macd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
            feature_all['macd_positive_ratio_' + str(day_cnt)] = feature_all['macd_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
         feature_all['macd_dif_dea'] = (tmp['macd_dif']-tmp['macd_dea']).groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1)))

         # boll线
         # 效果很好，可以多做几天的，比如3天，5天，10天，40天
         for day_cnt in [2, 3, 5, 10, 20, 40]:
            tmp['mb_' + str(day_cnt)] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            tmp['md_' + str(day_cnt)] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).std())
            tmp['up_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] + 2 * tmp['md_' + str(day_cnt)]
            tmp['dn_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] - 2 * tmp['md_' + str(day_cnt)]
            feature_all['width_' + str(day_cnt)] = (4 * tmp['md_' + str(day_cnt)] / tmp['mb_' + str(day_cnt)]).apply(lambda x: max(x, -3) if x<0 else min(x, 3))
            feature_all['close_mb' + str(day_cnt) + '_diff'] = ((tmp['close'] - tmp['mb_' + str(day_cnt)])/(2 * tmp['md_' + str(day_cnt)])).apply(lambda x: max(x, -3) if x<0 else min(x, 3))

         # cr指标
         # 似乎对中长期的指数cr指标效果较好，重新设计
         tmp = raw_k_data[['close', 'open', 'high', 'low']]
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
            feature_all['cr_bias_' + str(day_cnt) + 'd'] = (tmp['cr_' + str(day_cnt) + 'd']/tmp['cr_a_' + str(day_cnt) + 'd']).apply(lambda x: max(x, -3) if x<0 else min(x, 3))
            feature_all['cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: float2Bucket(float(x)*100, 0.1, 0, 300, 30))

            tmp['cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
            tmp['cr_a_' + str(day_cnt) + 'd'] = tmp['cr_a_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
            tmp['cr_b_' + str(day_cnt) + 'd'] = tmp['cr_b_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
            tmp['cr_c_' + str(day_cnt) + 'd'] = tmp['cr_c_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
            tmp['cr_d_' + str(day_cnt) + 'd'] = tmp['cr_d_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
            # bucket空间可以设置成 75万，多个特征可以共享embedding
            feature_all['cr_trend_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].str.cat([tmp['cr_a_' + str(day_cnt) + 'd'],tmp['cr_b_' + str(day_cnt) + 'd'],tmp['cr_c_' + str(day_cnt) + 'd'],tmp['cr_d_' + str(day_cnt) + 'd']], sep='_').apply(lambda x: self.tools.hash_bucket(x, 750000))
            for day_cnt_new in range(4):
               feature_all['cr_trend_' + str(day_cnt) + 'd' + '_' + str(day_cnt_new)] = feature_all['cr_trend_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt_new+2, window=day_cnt_new+2, center=False).apply(lambda y: y[0]))
         # rsi指标
         tmp = raw_k_data[['close', 'preclose']]
         tmp['price_dif'] = tmp['close'] - tmp['preclose']
         tmp['rsi_positive'] = tmp['price_dif'].apply(lambda x: max(x, 0))
         tmp['rsi_all'] = tmp['price_dif'].apply(lambda x: abs(x))
         for day_cnt in (3, 5, 10, 20, 40):
            feature_all['rsi_' + str(day_cnt) + 'd'] = (tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).sum()) / tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).sum()))

         day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
         for index in range(len(day_cnt_list)):
            day_cnt = day_cnt_list[index]
            day_cnt_last = day_cnt_list[index-1]
            feature_all['turn_' + str(day_cnt) + 'd' + '_avg'] = raw_k_data.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            feature_all['turn_rank_' + str(day_cnt) + 'd'] = feature_all['turn_' + str(day_cnt) + 'd' + '_avg']
            feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'] = raw_k_data['turn']/feature_all['turn_' + str(day_cnt) + 'd' + '_avg']
            feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + '_avg']/raw_k_data.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).max())
            feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif'] = raw_k_data.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).min())/feature_all['turn_' + str(day_cnt) + 'd' + '_avg']
            if index>0:
               feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = feature_all['turn_' + str(day_cnt_last) + 'd' + '_avg']/feature_all['turn_' + str(day_cnt) + 'd' + '_avg']
               feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['turn_' + str(day_cnt_last) + 'd' + 'max_dif']/feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif']
               feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'] = feature_all['turn_' + str(day_cnt_last) + 'd' + 'min_dif']/feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif']

            raw_k_data['close_' + str(day_cnt) + 'd' + '_avg'] = raw_k_data.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
            feature_all['close_' + str(day_cnt) + 'd' + 'avg_dif'] = raw_k_data['close']/raw_k_data['close_' + str(day_cnt) + 'd' + '_avg']
            feature_all['close_' + str(day_cnt) + 'd' + 'max_dif'] = raw_k_data['close']/raw_k_data.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).max())
            feature_all['close_' + str(day_cnt) + 'd' + 'min_dif'] = raw_k_data['close']/raw_k_data.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).min())
            feature_all['close_' + str(day_cnt) + 'd' + '_dif'] = raw_k_data.groupby(level=0)['close'].apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 1.0*(y[day_cnt-1])/y[0]))
            if index>0:
               feature_all['close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = raw_k_data['close_' + str(day_cnt_last) + 'd' + '_avg']/raw_k_data['close_' + str(day_cnt) + 'd' + '_avg']

         for index in range(len(day_cnt_list)):
            day_cnt = day_cnt_list[index]
            day_cnt_last = day_cnt_list[index - 1]
            if index > 0:
               feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 20, 0, 10, 200))
               feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 20, 0, 10, 200))
               feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'] = feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 20, 0, 10, 200))
               feature_all['close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = feature_all['close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 4, 200))
            feature_all['turn_' + str(day_cnt) + 'd' + '_avg'] = feature_all['turn_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: min(2, round(x/100, 4)))
            feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
            feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: round(x, 4))
            feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: round(x, 4))

            feature_all['close_' + str(day_cnt) + 'd' + 'avg_dif'] = feature_all['close_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
            feature_all['close_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['close_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 100, 0, 1, 100))
            feature_all['close_' + str(day_cnt) + 'd' + 'min_dif'] = feature_all['close_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 1, 10, 500))
            feature_all['close_' + str(day_cnt) + 'd' + '_dif'] = feature_all['close_' + str(day_cnt) + 'd' + '_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

         # 还需要再做一些量价类特征，比如高位放量，低位缩量等等
         for day_cnt in [3, 5, 10, 20, 30, 60, 120, 240]:
            raw_k_data['max_turn_index' + str(day_cnt) + 'd'] = raw_k_data['turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: y.argmax()))
            raw_k_data['max_close_index' + str(day_cnt) + 'd'] = raw_k_data['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: y.argmax()))
            raw_k_data['min_close_index' + str(day_cnt) + 'd'] = raw_k_data['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: y.argmin()))
            raw_k_data = raw_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'max_turn_index' + str(day_cnt) + 'd', 'close', 'max_turn_close' + str(day_cnt) + 'd', day_cnt))
            raw_k_data = raw_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'max_close_index' + str(day_cnt) + 'd', 'turn', 'max_closs_turn' + str(day_cnt) + 'd', day_cnt))
            raw_k_data = raw_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'min_close_index' + str(day_cnt) + 'd', 'turn', 'min_closs_turn' + str(day_cnt) + 'd', day_cnt))

            feature_all['max_turn_index' + str(day_cnt) + 'd'] = raw_k_data['max_turn_index' + str(day_cnt) + 'd']
            feature_all['max_close_index' + str(day_cnt) + 'd'] = raw_k_data['max_close_index' + str(day_cnt) + 'd']
            feature_all['min_close_index' + str(day_cnt) + 'd'] = raw_k_data['max_close_index' + str(day_cnt) + 'd']
            feature_all['max_turn_close' + str(day_cnt) + 'd'] = (raw_k_data['close']/raw_k_data['max_turn_close' + str(day_cnt) + 'd']).map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
            feature_all['max_closs_turn' + str(day_cnt) + 'd'] = (raw_k_data['max_closs_turn' + str(day_cnt) + 'd']/raw_k_data['turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
            feature_all['min_closs_turn' + str(day_cnt) + 'd'] = (raw_k_data['min_closs_turn' + str(day_cnt) + 'd']/raw_k_data['turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))

         # 新增过去n天每个涨幅段的天数比例，换手率的比例,涨停的天数,主要捕捉股性特征，判断股票是否活跃
         tmp = raw_k_data[['pctChg', 'turn', 'close', 'high', 'low', 'isST']]
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
            feature_all['pctChg_up_limit_' + str(day_cnt) + 'd'] = tmp['pctChg_up_limit'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())
            feature_all['pctChg_down_limit_' + str(day_cnt) + 'd'] = tmp['pctChg_down_limit'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())

            feature_all['pctChg_greater_3_' + str(day_cnt) + 'd'] = tmp['pctChg_greater_3'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())
            feature_all['pctChg_greater_6_' + str(day_cnt) + 'd'] = tmp['pctChg_greater_6'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())
            feature_all['pctChg_greater_9_' + str(day_cnt) + 'd'] = tmp['pctChg_greater_9'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())
            feature_all['pctChg_greater_13_' + str(day_cnt) + 'd'] = tmp['pctChg_greater_13'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())

            feature_all['pctChg_less_3_' + str(day_cnt) + 'd'] = tmp['pctChg_less_3'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())
            feature_all['pctChg_less_6_' + str(day_cnt) + 'd'] = tmp['pctChg_less_6'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())
            feature_all['pctChg_less_9_' + str(day_cnt) + 'd'] = tmp['pctChg_less_9'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())
            feature_all['pctChg_less_13_' + str(day_cnt) + 'd'] = tmp['pctChg_less_13'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())

            feature_all['turn_greater_3_' + str(day_cnt) + 'd'] = tmp['turn_greater_3'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())
            feature_all['turn_greater_6_' + str(day_cnt) + 'd'] = tmp['turn_greater_6'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())
            feature_all['turn_greater_10_' + str(day_cnt) + 'd'] = tmp['turn_greater_10'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())
            feature_all['turn_greater_15_' + str(day_cnt) + 'd'] = tmp['turn_greater_15'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())
            feature_all['turn_greater_21_' + str(day_cnt) + 'd'] = tmp['turn_greater_21'].groupby(level=0).apply(lambda x: x.rolling(min_periods=3, window=day_cnt, center=False).mean())

         # 新增过去n天股票的涨幅，已经在全市场的排名
         del tmp
         gc.collect()
         tmp = raw_k_data[['turn', 'close']]
         for day_cnt in [3, 5, 10, 20, 30, 60, 120]:
            feature_all['pctChg_' + str(day_cnt) + 'd'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: (y[day_cnt-1]-y[0])/y[0]))

         # 新增过去n天涨幅超过大盘，行业指数m个点的天数
         # 新增板块的最近n天的涨停股票数量，主要评估板块热度
         feature_all = feature_all.reset_index(level=0, drop=False)
         feature_all = feature_all.reset_index(level=0, drop=False)
         feature_all = feature_all.sort_values(['date', 'code'])
         feature_all = feature_all[feature_all['date'] > str(self.year)]

         del tmp
         gc.collect()
         tmp = feature_all[['date',"industry_id_level1","industry_id_level2","industry_id_level3"]]
         for day_cnt in [3, 5, 10, 20, 30, 60, 120]:
            tmp['pctChg_' + str(day_cnt) + 'd'] = feature_all['pctChg_' + str(day_cnt) + 'd']
            tmp['pctChg_rank' + str(day_cnt) + 'd'] = tmp['pctChg_' + str(day_cnt) + 'd'].groupby(tmp['date']).rank(ascending=False)

            tmp['pctChg_rank_industry1_' + str(day_cnt) + 'd'] = tmp.sort_values(['pctChg_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level1']).cumcount()+1
            tmp['pctChg_rank_industry2_' + str(day_cnt) + 'd'] = tmp.sort_values(['pctChg_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level2']).cumcount()+1
            tmp['pctChg_rank_industry3_' + str(day_cnt) + 'd'] = tmp.sort_values(['pctChg_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level3']).cumcount()+1

            if day_cnt == 3:
               tmp['code_count'] = tmp.groupby('date')['pctChg_rank' + str(day_cnt) + 'd'].transform('max')
               tmp['industry1_count'] = tmp.groupby(['date','industry_id_level1'])['pctChg_rank_industry1_' + str(day_cnt) + 'd'].transform('max')
               tmp['industry2_count'] = tmp.groupby(['date','industry_id_level2'])['pctChg_rank_industry2_' + str(day_cnt) + 'd'].transform('max')
               tmp['industry3_count'] = tmp.groupby(['date','industry_id_level3'])['pctChg_rank_industry3_' + str(day_cnt) + 'd'].transform('max')

            feature_all['pctChg_rank_ratio' + str(day_cnt) + 'd'] = (tmp['pctChg_rank' + str(day_cnt) + 'd']/tmp['code_count'])
            feature_all['pctChg_rank_ratio_industry1_' + str(day_cnt) + 'd'] = (tmp['pctChg_rank_industry1_' + str(day_cnt) + 'd']/tmp['industry1_count'])
            feature_all['pctChg_rank_ratio_industry2_' + str(day_cnt) + 'd'] = (tmp['pctChg_rank_industry2_' + str(day_cnt) + 'd']/tmp['industry2_count'])
            feature_all['pctChg_rank_ratio_industry3_' + str(day_cnt) + 'd'] = (tmp['pctChg_rank_industry3_' + str(day_cnt) + 'd']/tmp['industry3_count'])

            tmp['turn_rank_' + str(day_cnt) + 'd'] = feature_all['turn_rank_' + str(day_cnt) + 'd']
            tmp['turn_rank_' + str(day_cnt) + 'd'] = tmp['turn_rank_' + str(day_cnt) + 'd'].groupby(tmp['date']).rank(ascending=False)

            tmp['turn_rank_industry1_' + str(day_cnt) + 'd'] = tmp.sort_values(['turn_rank_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level1']).cumcount()+1
            tmp['turn_rank_industry2_' + str(day_cnt) + 'd'] = tmp.sort_values(['turn_rank_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level2']).cumcount()+1
            tmp['turn_rank_industry3_' + str(day_cnt) + 'd'] = tmp.sort_values(['turn_rank_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level3']).cumcount()+1


            feature_all['turn_rank_' + str(day_cnt) + 'd'] = (tmp['turn_rank_' + str(day_cnt) + 'd']/tmp['code_count'])
            feature_all['turn_rank_industry1_' + str(day_cnt) + 'd'] = (tmp['turn_rank_industry1_' + str(day_cnt) + 'd']/tmp['code_count'])
            feature_all['turn_rank_industry2_' + str(day_cnt) + 'd'] = (tmp['turn_rank_industry2_' + str(day_cnt) + 'd']/tmp['code_count'])
            feature_all['turn_rank_industry3_' + str(day_cnt) + 'd'] = (tmp['turn_rank_industry3_' + str(day_cnt) + 'd']/tmp['code_count'])

         feature_all = feature_all.reset_index(level=0, drop=True)
         if is_predict:
            feature_all = feature_all[(feature_all['date'] >= self.date_start) & (feature_all['date'] <= self.date_end)]
         del tmp
         del raw_k_data
         gc.collect()
         feature_all = feature_all.round(5)
         if os.path.isfile(self.code_feature_output_path):
            feature_all.to_csv(self.code_feature_output_path, mode='a', header=False, index=False)
         else:
            feature_all.to_csv(self.code_feature_output_path, mode='w', header=True, index=False)

      if self.is_predict:
         feature_all['date'] = feature_all['date'].map(lambda x: int(str(x).replace('-', '')[:6]))
         feature_all = feature_all[feature_all['code_market'] != 0]
         feature_all['label_7'] = 0
         feature_all['pctChg_7'] = 0
         feature_all['sh_pctChg_7'] = 0
         feature_all['label_15'] = 0
         feature_all['pctChg_15'] = 0
         feature_all['sh_pctChg_15'] = 0
         feature_all.to_csv('E:/pythonProject/future/data/datafile/prediction_sample/{model_name}/prediction_sample_{date}.csv'.format(model_name=self.model_name, date=str(self.date_end)), mode='a',header=True, index=False, encoding='utf-8')
      return feature_all
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

   years = [2023]
   is_predict = True
   model_name = 'model_v12'
   date_start = '2023-04-26'
   date_end = '2023-04-26'
   # years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
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