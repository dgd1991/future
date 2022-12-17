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
   def __init__(self,  k_file_path, year, quarter_file_path, is_predict, date_start, date_end):
      self.k_file_path = k_file_path + str(year) + '.csv'
      self.k_file_path_his = k_file_path + str(year - 1) + '.csv'
      self.quarter_file_path = quarter_file_path
      self.year = year
      self.tools = Tools()
      self.is_predict = is_predict
      self.date_start = date_start
      self.date_end = date_end

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
      raw_k_data = pd.read_csv(self.k_file_path)
      raw_k_data_his = pd.read_csv(self.k_file_path_his)
      if is_predict:
         raw_k_data_his = raw_k_data_his[raw_k_data_his['date']>self.tools.get_recent_month_date(self.date_start, -14)]
      raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
      raw_k_data = raw_k_data[(raw_k_data['industry_id_level3'] > 0) | (raw_k_data['code'] == 'sh.000001') | (raw_k_data['code'] == 'sz.399001') | (raw_k_data['code'] == 'sz.399006')]
      del raw_k_data_his
      gc.collect()
      raw_k_data["tradestatus"] = pd.to_numeric(raw_k_data["tradestatus"], errors='coerce')
      raw_k_data["turn"] = pd.to_numeric(raw_k_data["turn"], errors='coerce')
      raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')

      raw_k_data = raw_k_data[(raw_k_data['tradestatus'] == 1) & (raw_k_data['turn'] > 0) & (raw_k_data['pctChg'] <= 20) & (raw_k_data['pctChg'] >= -20)]

      raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))
      raw_k_data['is_new'] = raw_k_data["pctChg"].groupby(level=0).apply(lambda x: x.rolling(min_periods=20, window=20, center=False).apply(lambda y: y[0]))
      raw_k_data = raw_k_data[raw_k_data['is_new'].map(lambda x: False if np.isnan(x) else True)]
      raw_k_data = raw_k_data.reset_index(level=0, drop=True)
      raw_k_data = raw_k_data.reset_index(level=0, drop=False)

      raw_k_data["open"] = pd.to_numeric(raw_k_data["open"], errors='coerce')
      raw_k_data["close"] = pd.to_numeric(raw_k_data["close"], errors='coerce')
      raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')
      raw_k_data["preclose"] = pd.to_numeric(raw_k_data["preclose"], errors='coerce')
      raw_k_data["high"] = pd.to_numeric(raw_k_data["high"], errors='coerce')
      raw_k_data["low"] = pd.to_numeric(raw_k_data["low"], errors='coerce')
      raw_k_data['date'] = pd.to_datetime(raw_k_data['date'])
      raw_k_data['open_ratio'] = ((raw_k_data['open'] - raw_k_data['preclose']) / raw_k_data['preclose']).round(5)
      raw_k_data['close_ratio'] = ((raw_k_data['close'] - raw_k_data['open']) / raw_k_data['open']).round(5)
      raw_k_data['high_ratio'] = ((raw_k_data['high'] - raw_k_data['preclose']) / raw_k_data['preclose']).round(5)
      raw_k_data['low_ratio'] = ((raw_k_data['low'] - raw_k_data['preclose']) / raw_k_data['preclose']).round(5)
      raw_k_data['pctChg'] = raw_k_data['pctChg'].map(lambda x: x/100.0).round(5)
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
      feature_all['open_ratio_7d_avg'] = raw_k_data.groupby(level=0)['open_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      feature_all['close_ratio_7d_avg'] = raw_k_data.groupby(level=0)['close_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      feature_all['high_ratio_7d_avg'] = raw_k_data.groupby(level=0)['high_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      feature_all['low_ratio_7d_avg'] = raw_k_data.groupby(level=0)['low_ratio'].apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)

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
         tmp['min'] = tmp['low'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
         tmp['max'] = tmp['high'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
         feature_all['rsv_' + str(day_cnt)] = ((tmp['close'] - tmp['min'])/(tmp['max'] - tmp['min'])).round(5)
         feature_all['k_value_' + str(day_cnt)] = feature_all['rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean()).round(5)
         feature_all['d_value_' + str(day_cnt)] = feature_all['k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean()).round(5)
         feature_all['j_value_' + str(day_cnt)] = (3 * feature_all['k_value_' + str(day_cnt)] - 2 * feature_all['d_value_' + str(day_cnt)]).round(5)
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
      tmp['macd_dif_max'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
      tmp['macd_dif_min'] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
      tmp['macd_dea_max'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).max())
      tmp['macd_dea_min'] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=120, center=False).min())
      tmp['macd'] = (tmp['macd_dif'] - tmp['macd_dea']) * 2
      feature_all['macd_positive'] = tmp['macd'].apply(lambda x: 1 if x>0 else 0)
      feature_all['macd_dif_ratio'] = ((tmp['macd_dif']-tmp['macd_dif_min'])/(tmp['macd_dif_max']-tmp['macd_dif_min'])).round(5)
      for day_cnt in [2, 3, 5, 10, 20, 40]:
         feature_all['macd_dif_' + str(day_cnt)] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         feature_all['macd_dea_' + str(day_cnt)] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         feature_all['macd_' + str(day_cnt)] = tmp['macd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         feature_all['macd_positive_ratio_' + str(day_cnt)] = feature_all['macd_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean()).round(5)
      feature_all['macd_dif_dea'] = (tmp['macd_dif']-tmp['macd_dea']).groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1)))

      # boll线
      # 效果很好，可以多做几天的，比如3天，5天，10天，40天
      for day_cnt in [2, 3, 5, 10, 20, 40]:
         tmp['mb_' + str(day_cnt)] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
         tmp['md_' + str(day_cnt)] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).std())
         tmp['up_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] + 2 * tmp['md_' + str(day_cnt)]
         tmp['dn_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] - 2 * tmp['md_' + str(day_cnt)]
         feature_all['width_' + str(day_cnt)] = (4 * tmp['md_' + str(day_cnt)] / tmp['mb_' + str(day_cnt)]).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)
         feature_all['close_mb' + str(day_cnt) + '_diff'] = ((tmp['close'] - tmp['mb_' + str(day_cnt)])/(2 * tmp['md_' + str(day_cnt)])).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)

      # cr指标
      # 似乎对中长期的指数cr指标效果较好，重新设计
      tmp = raw_k_data[['close', 'open', 'high', 'low']]
      tmp['cr_m'] = (tmp['close'] + tmp['open'] + tmp['high'] + tmp['low'])/4
      tmp['cr_ym'] = tmp['cr_m'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0]))
      tmp['cr_p1_day'] = (tmp['high'] - tmp['cr_ym']).apply(lambda x: max(x, 0))
      tmp['cr_p2_day'] = (tmp['cr_ym'] - tmp['low']).apply(lambda x: max(x, 0))
      for day_cnt in [26]:
         tmp['cr_p1_' + str(day_cnt) + 'd'] = tmp['cr_p1_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         tmp['cr_p2_' + str(day_cnt) + 'd'] = tmp['cr_p2_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         tmp['cr_' + str(day_cnt) + 'd'] = tmp['cr_p1_' + str(day_cnt) + 'd'] / tmp['cr_p2_' + str(day_cnt) + 'd']
         tmp['cr_a_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=5, center=False).mean())
         tmp['cr_b_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=10, center=False).mean())
         tmp['cr_c_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).mean())
         tmp['cr_d_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=60, center=False).mean())
         feature_all['cr_bias_' + str(day_cnt) + 'd'] = (tmp['cr_' + str(day_cnt) + 'd']/tmp['cr_a_' + str(day_cnt) + 'd']).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)
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
         feature_all['rsi_' + str(day_cnt) + 'd'] = (tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum()) / tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())).round(5)

      day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
      for index in range(len(day_cnt_list)):
         day_cnt = day_cnt_list[index]
         day_cnt_last = day_cnt_list[index-1]
         feature_all['turn_' + str(day_cnt) + 'd' + '_avg'] = raw_k_data.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
         feature_all['turn_' + str(day_cnt) + 'd' + 'avg_dif'] = raw_k_data['turn']/feature_all['turn_' + str(day_cnt) + 'd' + '_avg']
         feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['turn_' + str(day_cnt) + 'd' + '_avg']/raw_k_data.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
         feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif'] = raw_k_data.groupby(level=0)['turn'].apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())/feature_all['turn_' + str(day_cnt) + 'd' + '_avg']
         if index>0:
            feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = feature_all['turn_' + str(day_cnt_last) + 'd' + '_avg']/feature_all['turn_' + str(day_cnt) + 'd' + '_avg']
            feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'] = feature_all['turn_' + str(day_cnt_last) + 'd' + 'max_dif']/feature_all['turn_' + str(day_cnt) + 'd' + 'max_dif']
            feature_all['turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'] = feature_all['turn_' + str(day_cnt_last) + 'd' + 'min_dif']/feature_all['turn_' + str(day_cnt) + 'd' + 'min_dif']

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
         raw_k_data['max_turn_index' + str(day_cnt) + 'd'] = raw_k_data['turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmax()))
         raw_k_data['max_close_index' + str(day_cnt) + 'd'] = raw_k_data['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmax()))
         raw_k_data['min_close_index' + str(day_cnt) + 'd'] = raw_k_data['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmin()))
         raw_k_data = raw_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'max_turn_index' + str(day_cnt) + 'd', 'close', 'max_turn_close' + str(day_cnt) + 'd', day_cnt))
         raw_k_data = raw_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'max_close_index' + str(day_cnt) + 'd', 'turn', 'max_closs_turn' + str(day_cnt) + 'd', day_cnt))
         raw_k_data = raw_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'min_close_index' + str(day_cnt) + 'd', 'turn', 'min_closs_turn' + str(day_cnt) + 'd', day_cnt))

         feature_all['max_turn_index' + str(day_cnt) + 'd'] = raw_k_data['max_turn_index' + str(day_cnt) + 'd']
         feature_all['max_close_index' + str(day_cnt) + 'd'] = raw_k_data['max_close_index' + str(day_cnt) + 'd']
         feature_all['min_close_index' + str(day_cnt) + 'd'] = raw_k_data['max_close_index' + str(day_cnt) + 'd']
         feature_all['max_turn_close' + str(day_cnt) + 'd'] = (raw_k_data['close']/raw_k_data['max_turn_close' + str(day_cnt) + 'd']).map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
         feature_all['max_closs_turn' + str(day_cnt) + 'd'] = (raw_k_data['max_closs_turn' + str(day_cnt) + 'd']/raw_k_data['turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
         feature_all['min_closs_turn' + str(day_cnt) + 'd'] = (raw_k_data['min_closs_turn' + str(day_cnt) + 'd']/raw_k_data['turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))

      feature_all = feature_all.reset_index(level=0, drop=False)
      feature_all = feature_all.reset_index(level=0, drop=False)
      feature_all = feature_all.sort_values(['date', 'code'])
      feature_all = feature_all[feature_all['date'] > str(self.year)]
      if is_predict:
         feature_all = feature_all[(feature_all['date'] >= self.date_start) & (feature_all['date'] <= self.date_end)]
      del tmp
      del raw_k_data

      # 一级行业特征
      raw_k_data = pd.read_csv(self.k_file_path)
      raw_k_data_his = pd.read_csv(self.k_file_path_his)
      if is_predict:
         raw_k_data_his = raw_k_data_his[raw_k_data_his['date']>self.tools.get_recent_month_date(self.date_start, -14)]
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
      raw_k_data_tmp = raw_k_data[["date","industry_id_level1", "market_value"]].groupby(["industry_id_level1","date"]).sum()
      raw_k_data_tmp.columns = ['industry_id_level1_market_value']
      raw_k_data_tmp['industry_id_level1_rise_ratio'] = raw_k_data[["date","industry_id_level1", "rise"]].groupby(["industry_id_level1","date"]).mean()
      raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
      raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
      raw_k_data = pd.merge(raw_k_data, raw_k_data_tmp, how="left", left_on=["date","industry_id_level1"], right_on=["date","industry_id_level1"])
      raw_k_data['market_value_ratio'] = raw_k_data["market_value"]/raw_k_data["industry_id_level1_market_value"]

      raw_k_data["turn"] = raw_k_data["turn"]*raw_k_data['market_value_ratio']
      raw_k_data['open_ratio'] = ((raw_k_data['open'] - raw_k_data['preclose']) / raw_k_data['preclose'])*raw_k_data['market_value_ratio']
      raw_k_data['close_ratio'] = ((raw_k_data['close'] - raw_k_data['open']) / raw_k_data['open'])*raw_k_data['market_value_ratio']
      raw_k_data['high_ratio'] = ((raw_k_data['high'] - raw_k_data['preclose']) / raw_k_data['preclose'])*raw_k_data['market_value_ratio']
      raw_k_data['low_ratio'] = ((raw_k_data['low'] - raw_k_data['preclose']) / raw_k_data['preclose'])*raw_k_data['market_value_ratio']
      raw_k_data['pctChg'] = raw_k_data['pctChg']*raw_k_data['market_value_ratio']
      raw_k_data_tmp['industry_id_level1_rise_ratio'] = raw_k_data_tmp['industry_id_level1_rise_ratio']*raw_k_data['market_value_ratio']

      raw_k_data["peTTM"] = raw_k_data["peTTM"]*raw_k_data['market_value_ratio']
      raw_k_data["pcfNcfTTM"] = raw_k_data["pcfNcfTTM"]*raw_k_data['market_value_ratio']
      raw_k_data["pbMRQ"] = raw_k_data["pbMRQ"]*raw_k_data['market_value_ratio']
      raw_k_data["industry_id_level1_rise_ratio"] = raw_k_data["industry_id_level1_rise_ratio"]*raw_k_data['market_value_ratio']

      industry_id_level1_k_data = raw_k_data[["industry_id_level1","open_ratio","close_ratio","high_ratio","low_ratio","turn","date","pctChg","peTTM","pcfNcfTTM","pbMRQ", 'industry_id_level1_rise_ratio', 'market_value']].groupby(['industry_id_level1','date']).sum().round(5)
      industry_id_level1_k_data.columns = ["industry_id_level1_open_ratio","industry_id_level1_close_ratio","industry_id_level1_high_ratio","industry_id_level1_low_ratio","industry_id_level1_turn","industry_id_level1_pctChg","industry_id_level1_peTTM","industry_id_level1_pcfNcfTTM","industry_id_level1_pbMRQ", 'industry_id_level1_rise_ratio', 'industry_id_level1_market_value']
      del raw_k_data
      gc.collect()
      if os.path.isfile('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level1_' + str(year-1) + '.csv'):
          industry_id_level1_k_data_his = pd.read_csv('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level1_' + str(year-1) + '.csv')
          industry_id_level1_k_data_his = industry_id_level1_k_data_his[['industry_id_level1','date','industry_id_level1_close']]
          industry_id_level1_k_data_his['date'] = pd.to_datetime(industry_id_level1_k_data_his['date'])
      industry_id_level1_k_data = industry_id_level1_k_data.reset_index(level=0, drop=False)
      industry_id_level1_k_data = industry_id_level1_k_data.reset_index(level=0, drop=False)
      if os.path.isfile('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level1_' + str(year-1) + '.csv'):
         industry_id_level1_k_data = pd.merge(industry_id_level1_k_data, industry_id_level1_k_data_his, how="left", left_on=['industry_id_level1','date'], right_on=['industry_id_level1','date'])
      else:
         industry_id_level1_k_data['industry_id_level1_close'] = industry_id_level1_k_data['industry_id_level1_open_ratio'].apply(lambda x: 1000)

      # industry_id_level1_k_data = industry_id_level1_k_data.set_index('industry_id_level1', drop=True)
      # industry_id_level1_k_data = industry_id_level1_k_data.set_index('date', drop=True)
      industry_id_level1_k_data = industry_id_level1_k_data.groupby('industry_id_level1').apply(lambda x: x.set_index('date', drop=True))

      industry_id_level1_k_data = industry_id_level1_k_data.groupby(level=0).apply(lambda x: self.func(x, 'industry_id_level1_pctChg', 'industry_id_level1_close')).round(5)
      industry_id_level1_k_data['industry_id_level1_preclose'] = industry_id_level1_k_data['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0])).round(5)
      industry_id_level1_k_data['industry_id_level1_open'] = industry_id_level1_k_data['industry_id_level1_preclose']*(industry_id_level1_k_data['industry_id_level1_open_ratio'].apply(lambda x: x + 1)).round(5)
      industry_id_level1_k_data['industry_id_level1_high'] = industry_id_level1_k_data['industry_id_level1_preclose']*(industry_id_level1_k_data['industry_id_level1_high_ratio'].apply(lambda x: x + 1)).round(5)
      industry_id_level1_k_data['industry_id_level1_low'] = industry_id_level1_k_data['industry_id_level1_preclose']*(industry_id_level1_k_data['industry_id_level1_low_ratio'].apply(lambda x: x + 1)).round(5)

      # 写出指数点数
      industry_id_level1_k_data_out = industry_id_level1_k_data[['industry_id_level1_open', 'industry_id_level1_close', 'industry_id_level1_high', 'industry_id_level1_low']]
      industry_id_level1_k_data_out = industry_id_level1_k_data_out.reset_index(level=0, drop=False)
      industry_id_level1_k_data_out = industry_id_level1_k_data_out.reset_index(level=0, drop=False)
      industry_id_level1_k_data_out.to_csv('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level1_' + str(year) + '.csv', mode='w', header=True, index=False)
      del industry_id_level1_k_data_out
      gc.collect()

      industry_id_level1_k_data['industry_id_level1_open_ratio_7d_avg'] = industry_id_level1_k_data['industry_id_level1_open_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level1_k_data['industry_id_level1_close_ratio_7d_avg'] = industry_id_level1_k_data['industry_id_level1_close_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level1_k_data['industry_id_level1_high_ratio_7d_avg'] = industry_id_level1_k_data['industry_id_level1_high_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level1_k_data['industry_id_level1_low_ratio_7d_avg'] = industry_id_level1_k_data['industry_id_level1_low_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level1_k_data['industry_id_level1_turn'] = industry_id_level1_k_data["industry_id_level1_turn"].map(lambda x: float2Bucket(float(x) * 100, 1, 0, 50, 50))

      industry_id_level1_k_data['industry_id_level1_peTTM'] = industry_id_level1_k_data["industry_id_level1_peTTM"].map(lambda x: float2Bucket(float(x) + 200, 0.5, 0, 400, 200))
      industry_id_level1_k_data['industry_id_level1_pcfNcfTTM'] = industry_id_level1_k_data["industry_id_level1_pcfNcfTTM"].map(lambda x: float2Bucket(float(x) + 200, 0.5, 0, 400, 200))
      industry_id_level1_k_data['industry_id_level1_pbMRQ'] = industry_id_level1_k_data["industry_id_level1_pbMRQ"].map(lambda x: float2Bucket(float(x), 2, 0, 100, 200))
      industry_id_level1_k_data['industry_id_level1_market_value'] = industry_id_level1_k_data["industry_id_level1_market_value"].map(lambda x: None if x == '' else bignumber2Bucket(float(x), 1.25, 60))

      # kdj 5, 9, 19, 36, 45, 73，
      # 任意初始化，超过30天后的kdj值基本一样
      tmp = industry_id_level1_k_data[["industry_id_level1_low", "industry_id_level1_high", "industry_id_level1_close"]]
      for day_cnt in [5, 9, 19, 73]:
      # industry_id_level1_k_data['industry_id_level1_rsv'] = industry_id_level1_k_data[["industry_id_level1_low", "industry_id_level1_high", "industry_id_level1_close"]].groupby(level=0).apply(lambda x: (x.close-x.low.rolling(min_periods=1, window=day_cnt, center=False).min())/(x.high.rolling(min_periods=1, window=day_cnt, center=False).max()-x.low.rolling(min_periods=1, window=day_cnt, center=False).min()))
         tmp['min'] = tmp["industry_id_level1_low"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
         tmp['max'] = tmp["industry_id_level1_high"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
         industry_id_level1_k_data['industry_id_level1_rsv_' + str(day_cnt)] = ((tmp["industry_id_level1_close"] - tmp['min'])/(tmp['max'] - tmp['min'])).round(5)
         industry_id_level1_k_data['industry_id_level1_k_value_' + str(day_cnt)] = industry_id_level1_k_data['industry_id_level1_rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean()).round(5)
         industry_id_level1_k_data['industry_id_level1_d_value_' + str(day_cnt)] = industry_id_level1_k_data['industry_id_level1_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean()).round(5)
         industry_id_level1_k_data['industry_id_level1_j_value_' + str(day_cnt)] = (3 * industry_id_level1_k_data['industry_id_level1_k_value_' + str(day_cnt)] - 2 * industry_id_level1_k_data['industry_id_level1_d_value_' + str(day_cnt)]).round(5)
         # industry_id_level1_k_data['industry_id_level1_k_value_trend_' + str(day_cnt)] = industry_id_level1_k_data['industry_id_level1_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1])-y[0]))
         # industry_id_level1_k_data['industry_id_level1_kd_value' + str(day_cnt)] = (industry_id_level1_k_data['industry_id_level1_k_value_' + str(day_cnt)] - industry_id_level1_k_data['industry_id_level1_d_value_' + str(day_cnt)]).rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1))

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
      industry_id_level1_k_data['industry_id_level1_macd_dif_ratio'] = ((tmp['macd_dif']-tmp['macd_dif_min'])/(tmp['macd_dif_max']-tmp['macd_dif_min'])).round(5)
      for day_cnt in [2, 3, 5, 10, 20, 40]:
         industry_id_level1_k_data['industry_id_level1_macd_dif_' + str(day_cnt)] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         industry_id_level1_k_data['industry_id_level1_macd_dea_' + str(day_cnt)] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         industry_id_level1_k_data['industry_id_level1_macd_' + str(day_cnt)] = tmp['macd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         industry_id_level1_k_data['industry_id_level1_macd_positive_ratio_' + str(day_cnt)] = industry_id_level1_k_data['industry_id_level1_macd_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean()).round(5)
      industry_id_level1_k_data['industry_id_level1_macd_dif_dea'] = (tmp['macd_dif']-tmp['macd_dea']).groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1)))

      # boll线
      for day_cnt in [2, 3, 5, 10, 20, 40]:
         tmp['mb_' + str(day_cnt)] = tmp['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
         tmp['md_' + str(day_cnt)] = tmp['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).std())
         tmp['up_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] + 2 * tmp['md_' + str(day_cnt)]
         tmp['dn_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] - 2 * tmp['md_' + str(day_cnt)]
         industry_id_level1_k_data['industry_id_level1_width_' + str(day_cnt)] = (4 * tmp['md_' + str(day_cnt)] / tmp['mb_' + str(day_cnt)]).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)
         industry_id_level1_k_data['industry_id_level1_close_mb' + str(day_cnt) + '_diff'] = ((tmp['industry_id_level1_close'] - tmp['mb_' + str(day_cnt)])/(2 * tmp['md_' + str(day_cnt)])).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)

      # cr指标
      tmp = industry_id_level1_k_data[["industry_id_level1_close", "industry_id_level1_open", "industry_id_level1_high", "industry_id_level1_low"]]
      tmp['cr_m'] = (tmp["industry_id_level1_close"] + tmp["industry_id_level1_open"] + tmp["industry_id_level1_high"] + tmp["industry_id_level1_low"])/4
      tmp['cr_ym'] = tmp['cr_m'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0]))
      tmp['cr_p1_day'] = (tmp['industry_id_level1_high'] - tmp['cr_ym']).apply(lambda x: max(x, 0))
      tmp['cr_p2_day'] = (tmp['cr_ym'] - tmp['industry_id_level1_low']).apply(lambda x: max(x, 0))
      for day_cnt in [26]:
         tmp['cr_p1_' + str(day_cnt) + 'd'] = tmp['cr_p1_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         tmp['cr_p2_' + str(day_cnt) + 'd'] = tmp['cr_p2_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         tmp['cr_' + str(day_cnt) + 'd'] = tmp['cr_p1_' + str(day_cnt) + 'd'] / tmp['cr_p2_' + str(day_cnt) + 'd']
         tmp['cr_a_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=5, center=False).mean())
         tmp['cr_b_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=10, center=False).mean())
         tmp['cr_c_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).mean())
         tmp['cr_d_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=60, center=False).mean())
         industry_id_level1_k_data['industry_id_level1_cr_bias_' + str(day_cnt) + 'd'] = (tmp['cr_' + str(day_cnt) + 'd']/tmp['cr_a_' + str(day_cnt) + 'd']).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)
         industry_id_level1_k_data['industry_id_level1_cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: float2Bucket(float(x)*100, 0.1, 0, 300, 30))

         tmp['cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_a_' + str(day_cnt) + 'd'] = tmp['cr_a_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_b_' + str(day_cnt) + 'd'] = tmp['cr_b_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_c_' + str(day_cnt) + 'd'] = tmp['cr_c_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_d_' + str(day_cnt) + 'd'] = tmp['cr_d_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         # bucket空间可以设置成 75万，多个特征可以共享embedding
         industry_id_level1_k_data['industry_id_level1_cr_trend_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].str.cat([tmp['cr_a_' + str(day_cnt) + 'd'],tmp['cr_b_' + str(day_cnt) + 'd'],tmp['cr_c_' + str(day_cnt) + 'd'],tmp['cr_d_' + str(day_cnt) + 'd']], sep='_').apply(lambda x: self.tools.hash_bucket(x, 750000))
         for day_cnt_new in range(4):
            industry_id_level1_k_data['industry_id_level1_cr_trend_' + str(day_cnt) + 'd' + '_' + str(day_cnt_new)] = industry_id_level1_k_data['industry_id_level1_cr_trend_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt_new+2, window=day_cnt_new+2, center=False).apply(lambda y: y[0]))
      # rsi指标
      tmp = industry_id_level1_k_data[["industry_id_level1_close", "industry_id_level1_preclose"]]
      tmp['price_dif'] = tmp["industry_id_level1_close"] - tmp["industry_id_level1_preclose"]
      tmp['rsi_positive'] = tmp['price_dif'].apply(lambda x: max(x, 0))
      tmp['rsi_all'] = tmp['price_dif'].apply(lambda x: abs(x))
      for day_cnt in (3, 5, 10, 20, 40):
         tmp['rsi_positive_sum'] = tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         tmp['rsi_all_sum'] = tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         industry_id_level1_k_data['industry_id_level1_rsi_' + str(day_cnt) + 'd'] =  (tmp['rsi_positive_sum'] / tmp['rsi_all_sum']).round(5)

      tmp = industry_id_level1_k_data[["industry_id_level1_turn", "industry_id_level1_close"]]
      day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
      for index in range(len(day_cnt_list)):
         day_cnt = day_cnt_list[index]
         day_cnt_last = day_cnt_list[index-1]
         industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level1_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
         industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = tmp['industry_id_level1_turn']/industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg']
         industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg']/tmp['industry_id_level1_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
         industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'min_dif'] = tmp['industry_id_level1_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())/industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg']
         if index>0:
            industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + 'd' + '_avg']/industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg']
            industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + 'd' + 'max_dif']/industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'max_dif']
            industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + 'd' + 'min_dif']/industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'min_dif']

         tmp['industry_id_level1_close_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
         industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'avg_dif'] = tmp['industry_id_level1_close']/tmp['industry_id_level1_close_' + str(day_cnt) + 'd' + '_avg']
         industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'max_dif'] = tmp['industry_id_level1_close']/tmp['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
         industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'min_dif'] = tmp['industry_id_level1_close']/tmp['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
         industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + '_dif'] = tmp['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 1.0*(y[day_cnt-1])/y[0]))
         if index>0:
            industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level1_close_' + str(day_cnt_last) + 'd' + '_avg']/tmp['industry_id_level1_close_' + str(day_cnt) + 'd' + '_avg']

      for index in range(len(day_cnt_list)):
         day_cnt = day_cnt_list[index]
         day_cnt_last = day_cnt_list[index - 1]
         if index > 0:
            industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 2, 100))
            industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 2, 100))
            industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 2, 100))
            industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 4, 200))
         industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: min(2, round(x/100, 4)))
         industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
         industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: round(x, 4))
         industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level1_k_data['industry_id_level1_turn_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: round(x, 4))

         industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
         industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 100, 0, 1, 100))
         industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 1, 10, 500))
         industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + '_dif'] = industry_id_level1_k_data['industry_id_level1_close_' + str(day_cnt) + 'd' + '_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

      # 还需要再做一些量价类特征，比如高位放量，低位缩量等等
      for day_cnt in [3, 5, 10, 20, 30, 60, 120, 240]:
         industry_id_level1_k_data['industry_id_level1_max_turn_index' + str(day_cnt) + 'd'] = tmp['industry_id_level1_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmax()))
         industry_id_level1_k_data['industry_id_level1_max_close_index' + str(day_cnt) + 'd'] = tmp['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmax()))
         industry_id_level1_k_data['industry_id_level1_min_close_index' + str(day_cnt) + 'd'] = tmp['industry_id_level1_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmin()))
         industry_id_level1_k_data = industry_id_level1_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'industry_id_level1_max_turn_index' + str(day_cnt) + 'd', 'industry_id_level1_close', 'industry_id_level1_max_turn_close' + str(day_cnt) + 'd', day_cnt))
         industry_id_level1_k_data = industry_id_level1_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'industry_id_level1_max_close_index' + str(day_cnt) + 'd', 'industry_id_level1_turn', 'industry_id_level1_max_closs_turn' + str(day_cnt) + 'd', day_cnt))
         industry_id_level1_k_data = industry_id_level1_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'industry_id_level1_min_close_index' + str(day_cnt) + 'd', 'industry_id_level1_turn', 'industry_id_level1_min_closs_turn' + str(day_cnt) + 'd', day_cnt))

         industry_id_level1_k_data['industry_id_level1_max_turn_close' + str(day_cnt) + 'd'] = (tmp['industry_id_level1_close']/industry_id_level1_k_data['industry_id_level1_max_turn_close' + str(day_cnt) + 'd']).map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
         industry_id_level1_k_data['industry_id_level1_max_closs_turn' + str(day_cnt) + 'd'] = (industry_id_level1_k_data['industry_id_level1_max_closs_turn' + str(day_cnt) + 'd']/tmp['industry_id_level1_turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
         industry_id_level1_k_data['industry_id_level1_min_closs_turn' + str(day_cnt) + 'd'] = (industry_id_level1_k_data['industry_id_level1_min_closs_turn' + str(day_cnt) + 'd']/tmp['industry_id_level1_turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))

      industry_id_level1_k_data = industry_id_level1_k_data.reset_index(level=0, drop=True)
      industry_id_level1_k_data = industry_id_level1_k_data.reset_index(level=0, drop=False)
      selected_columns = ['date','industry_id_level1','industry_id_level1_open_ratio','industry_id_level1_close_ratio','industry_id_level1_high_ratio','industry_id_level1_low_ratio','industry_id_level1_turn','industry_id_level1_pctChg','industry_id_level1_peTTM','industry_id_level1_pcfNcfTTM','industry_id_level1_pbMRQ','industry_id_level1_rise_ratio','industry_id_level1_market_value','industry_id_level1_open_ratio_7d_avg','industry_id_level1_close_ratio_7d_avg','industry_id_level1_high_ratio_7d_avg','industry_id_level1_low_ratio_7d_avg','industry_id_level1_rsv_5','industry_id_level1_k_value_5','industry_id_level1_d_value_5','industry_id_level1_j_value_5','industry_id_level1_rsv_9','industry_id_level1_k_value_9','industry_id_level1_d_value_9','industry_id_level1_j_value_9','industry_id_level1_rsv_19','industry_id_level1_k_value_19','industry_id_level1_d_value_19','industry_id_level1_j_value_19','industry_id_level1_rsv_73','industry_id_level1_k_value_73','industry_id_level1_d_value_73','industry_id_level1_j_value_73','industry_id_level1_macd_positive','industry_id_level1_macd_dif_ratio','industry_id_level1_macd_dif_2','industry_id_level1_macd_dea_2','industry_id_level1_macd_2','industry_id_level1_macd_positive_ratio_2','industry_id_level1_macd_dif_3','industry_id_level1_macd_dea_3','industry_id_level1_macd_3','industry_id_level1_macd_positive_ratio_3','industry_id_level1_macd_dif_5','industry_id_level1_macd_dea_5','industry_id_level1_macd_5','industry_id_level1_macd_positive_ratio_5','industry_id_level1_macd_dif_10','industry_id_level1_macd_dea_10','industry_id_level1_macd_10','industry_id_level1_macd_positive_ratio_10','industry_id_level1_macd_dif_20','industry_id_level1_macd_dea_20','industry_id_level1_macd_20','industry_id_level1_macd_positive_ratio_20','industry_id_level1_macd_dif_40','industry_id_level1_macd_dea_40','industry_id_level1_macd_40','industry_id_level1_macd_positive_ratio_40','industry_id_level1_macd_dif_dea','industry_id_level1_width_2','industry_id_level1_close_mb2_diff','industry_id_level1_width_3','industry_id_level1_close_mb3_diff','industry_id_level1_width_5','industry_id_level1_close_mb5_diff','industry_id_level1_width_10','industry_id_level1_close_mb10_diff','industry_id_level1_width_20','industry_id_level1_close_mb20_diff','industry_id_level1_width_40','industry_id_level1_close_mb40_diff','industry_id_level1_cr_bias_26d','industry_id_level1_cr_26d','industry_id_level1_cr_trend_26d','industry_id_level1_cr_trend_26d_0','industry_id_level1_cr_trend_26d_1','industry_id_level1_cr_trend_26d_2','industry_id_level1_cr_trend_26d_3','industry_id_level1_rsi_3d','industry_id_level1_rsi_5d','industry_id_level1_rsi_10d','industry_id_level1_rsi_20d','industry_id_level1_rsi_40d','industry_id_level1_turn_3d_avg','industry_id_level1_turn_3davg_dif','industry_id_level1_turn_3dmax_dif','industry_id_level1_turn_3dmin_dif','industry_id_level1_close_3davg_dif','industry_id_level1_close_3dmax_dif','industry_id_level1_close_3dmin_dif','industry_id_level1_close_3d_dif','industry_id_level1_turn_5d_avg','industry_id_level1_turn_5davg_dif','industry_id_level1_turn_5dmax_dif','industry_id_level1_turn_5dmin_dif','industry_id_level1_turn_3_5d_avg','industry_id_level1_turn_3_5dmax_dif','industry_id_level1_turn_3_5dmin_dif','industry_id_level1_close_5davg_dif','industry_id_level1_close_5dmax_dif','industry_id_level1_close_5dmin_dif','industry_id_level1_close_5d_dif','industry_id_level1_close_3_5d_avg','industry_id_level1_turn_10d_avg','industry_id_level1_turn_10davg_dif','industry_id_level1_turn_10dmax_dif','industry_id_level1_turn_10dmin_dif','industry_id_level1_turn_5_10d_avg','industry_id_level1_turn_5_10dmax_dif','industry_id_level1_turn_5_10dmin_dif','industry_id_level1_close_10davg_dif','industry_id_level1_close_10dmax_dif','industry_id_level1_close_10dmin_dif','industry_id_level1_close_10d_dif','industry_id_level1_close_5_10d_avg','industry_id_level1_turn_20d_avg','industry_id_level1_turn_20davg_dif','industry_id_level1_turn_20dmax_dif','industry_id_level1_turn_20dmin_dif','industry_id_level1_turn_10_20d_avg','industry_id_level1_turn_10_20dmax_dif','industry_id_level1_turn_10_20dmin_dif','industry_id_level1_close_20davg_dif','industry_id_level1_close_20dmax_dif','industry_id_level1_close_20dmin_dif','industry_id_level1_close_20d_dif','industry_id_level1_close_10_20d_avg','industry_id_level1_turn_30d_avg','industry_id_level1_turn_30davg_dif','industry_id_level1_turn_30dmax_dif','industry_id_level1_turn_30dmin_dif','industry_id_level1_turn_20_30d_avg','industry_id_level1_turn_20_30dmax_dif','industry_id_level1_turn_20_30dmin_dif','industry_id_level1_close_30davg_dif','industry_id_level1_close_30dmax_dif','industry_id_level1_close_30dmin_dif','industry_id_level1_close_30d_dif','industry_id_level1_close_20_30d_avg','industry_id_level1_turn_60d_avg','industry_id_level1_turn_60davg_dif','industry_id_level1_turn_60dmax_dif','industry_id_level1_turn_60dmin_dif','industry_id_level1_turn_30_60d_avg','industry_id_level1_turn_30_60dmax_dif','industry_id_level1_turn_30_60dmin_dif','industry_id_level1_close_60davg_dif','industry_id_level1_close_60dmax_dif','industry_id_level1_close_60dmin_dif','industry_id_level1_close_60d_dif','industry_id_level1_close_30_60d_avg','industry_id_level1_turn_120d_avg','industry_id_level1_turn_120davg_dif','industry_id_level1_turn_120dmax_dif','industry_id_level1_turn_120dmin_dif','industry_id_level1_turn_60_120d_avg','industry_id_level1_turn_60_120dmax_dif','industry_id_level1_turn_60_120dmin_dif','industry_id_level1_close_120davg_dif','industry_id_level1_close_120dmax_dif','industry_id_level1_close_120dmin_dif','industry_id_level1_close_120d_dif','industry_id_level1_close_60_120d_avg','industry_id_level1_turn_240d_avg','industry_id_level1_turn_240davg_dif','industry_id_level1_turn_240dmax_dif','industry_id_level1_turn_240dmin_dif','industry_id_level1_turn_120_240d_avg','industry_id_level1_turn_120_240dmax_dif','industry_id_level1_turn_120_240dmin_dif','industry_id_level1_close_240davg_dif','industry_id_level1_close_240dmax_dif','industry_id_level1_close_240dmin_dif','industry_id_level1_close_240d_dif','industry_id_level1_close_120_240d_avg','industry_id_level1_max_turn_index3d','industry_id_level1_max_close_index3d','industry_id_level1_min_close_index3d','industry_id_level1_max_turn_close3d','industry_id_level1_max_closs_turn3d','industry_id_level1_min_closs_turn3d','industry_id_level1_max_turn_index5d','industry_id_level1_max_close_index5d','industry_id_level1_min_close_index5d','industry_id_level1_max_turn_close5d','industry_id_level1_max_closs_turn5d','industry_id_level1_min_closs_turn5d','industry_id_level1_max_turn_index10d','industry_id_level1_max_close_index10d','industry_id_level1_min_close_index10d','industry_id_level1_max_turn_close10d','industry_id_level1_max_closs_turn10d','industry_id_level1_min_closs_turn10d','industry_id_level1_max_turn_index20d','industry_id_level1_max_close_index20d','industry_id_level1_min_close_index20d','industry_id_level1_max_turn_close20d','industry_id_level1_max_closs_turn20d','industry_id_level1_min_closs_turn20d','industry_id_level1_max_turn_index30d','industry_id_level1_max_close_index30d','industry_id_level1_min_close_index30d','industry_id_level1_max_turn_close30d','industry_id_level1_max_closs_turn30d','industry_id_level1_min_closs_turn30d','industry_id_level1_max_turn_index60d','industry_id_level1_max_close_index60d','industry_id_level1_min_close_index60d','industry_id_level1_max_turn_close60d','industry_id_level1_max_closs_turn60d','industry_id_level1_min_closs_turn60d','industry_id_level1_max_turn_index120d','industry_id_level1_max_close_index120d','industry_id_level1_min_close_index120d','industry_id_level1_max_turn_close120d','industry_id_level1_max_closs_turn120d','industry_id_level1_min_closs_turn120d','industry_id_level1_max_turn_index240d','industry_id_level1_max_close_index240d','industry_id_level1_min_close_index240d','industry_id_level1_max_turn_close240d','industry_id_level1_max_closs_turn240d','industry_id_level1_min_closs_turn240d']
      industry_id_level1_k_data = industry_id_level1_k_data[selected_columns]

      feature_all = pd.merge(feature_all, industry_id_level1_k_data, how="left", left_on=["date",'industry_id_level1'],right_on=["date",'industry_id_level1'])
      feature_all = feature_all.sort_values(['date', 'code'])
      del industry_id_level1_k_data
      del tmp
      gc.collect()

      # 二级行业特征
      raw_k_data = pd.read_csv(self.k_file_path)
      raw_k_data_his = pd.read_csv(self.k_file_path_his)
      if is_predict:
         raw_k_data_his = raw_k_data_his[raw_k_data_his['date']>self.tools.get_recent_month_date(self.date_start, -14)]
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
      raw_k_data_tmp = raw_k_data[["date","industry_id_level2", "market_value"]].groupby(["industry_id_level2","date"]).sum()
      raw_k_data_tmp.columns = ['industry_id_level2_market_value']
      raw_k_data_tmp['industry_id_level2_rise_ratio'] = raw_k_data[["date","industry_id_level2", "rise"]].groupby(["industry_id_level2","date"]).mean()
      raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
      raw_k_data_tmp = raw_k_data_tmp.reset_index(level=0, drop=False)
      raw_k_data = pd.merge(raw_k_data, raw_k_data_tmp, how="left", left_on=["date","industry_id_level2"], right_on=["date","industry_id_level2"])
      raw_k_data['market_value_ratio'] = raw_k_data["market_value"]/raw_k_data["industry_id_level2_market_value"]

      raw_k_data["turn"] = raw_k_data["turn"]*raw_k_data['market_value_ratio']
      raw_k_data['open_ratio'] = ((raw_k_data['open'] - raw_k_data['preclose']) / raw_k_data['preclose'])*raw_k_data['market_value_ratio']
      raw_k_data['close_ratio'] = ((raw_k_data['close'] - raw_k_data['open']) / raw_k_data['open'])*raw_k_data['market_value_ratio']
      raw_k_data['high_ratio'] = ((raw_k_data['high'] - raw_k_data['preclose']) / raw_k_data['preclose'])*raw_k_data['market_value_ratio']
      raw_k_data['low_ratio'] = ((raw_k_data['low'] - raw_k_data['preclose']) / raw_k_data['preclose'])*raw_k_data['market_value_ratio']
      raw_k_data['pctChg'] = raw_k_data['pctChg']*raw_k_data['market_value_ratio']
      raw_k_data_tmp['industry_id_level2_rise_ratio'] = raw_k_data_tmp['industry_id_level2_rise_ratio']*raw_k_data['market_value_ratio']

      raw_k_data["peTTM"] = raw_k_data["peTTM"]*raw_k_data['market_value_ratio']
      raw_k_data["pcfNcfTTM"] = raw_k_data["pcfNcfTTM"]*raw_k_data['market_value_ratio']
      raw_k_data["pbMRQ"] = raw_k_data["pbMRQ"]*raw_k_data['market_value_ratio']
      raw_k_data["industry_id_level2_rise_ratio"] = raw_k_data["industry_id_level2_rise_ratio"]*raw_k_data['market_value_ratio']

      industry_id_level2_k_data = raw_k_data[["industry_id_level2","open_ratio","close_ratio","high_ratio","low_ratio","turn","date","pctChg","peTTM","pcfNcfTTM","pbMRQ", 'industry_id_level2_rise_ratio', 'market_value']].groupby(['industry_id_level2','date']).sum().round(5)
      industry_id_level2_k_data.columns = ["industry_id_level2_open_ratio","industry_id_level2_close_ratio","industry_id_level2_high_ratio","industry_id_level2_low_ratio","industry_id_level2_turn","industry_id_level2_pctChg","industry_id_level2_peTTM","industry_id_level2_pcfNcfTTM","industry_id_level2_pbMRQ", 'industry_id_level2_rise_ratio', 'industry_id_level2_market_value']
      del raw_k_data
      gc.collect()
      if os.path.isfile('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level2_' + str(year-1) + '.csv'):
          industry_id_level2_k_data_his = pd.read_csv('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level2_' + str(year-1) + '.csv')
          industry_id_level2_k_data_his = industry_id_level2_k_data_his[['industry_id_level2','date','industry_id_level2_close']]
          industry_id_level2_k_data_his['date'] = pd.to_datetime(industry_id_level2_k_data_his['date'])
      industry_id_level2_k_data = industry_id_level2_k_data.reset_index(level=0, drop=False)
      industry_id_level2_k_data = industry_id_level2_k_data.reset_index(level=0, drop=False)
      if os.path.isfile('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level2_' + str(year-1) + '.csv'):
         industry_id_level2_k_data = pd.merge(industry_id_level2_k_data, industry_id_level2_k_data_his, how="left", left_on=['industry_id_level2','date'], right_on=['industry_id_level2','date'])
      else:
         industry_id_level2_k_data['industry_id_level2_close'] = industry_id_level2_k_data['industry_id_level2_open_ratio'].apply(lambda x: 1000)

      # industry_id_level2_k_data = industry_id_level2_k_data.set_index('industry_id_level2', drop=True)
      # industry_id_level2_k_data = industry_id_level2_k_data.set_index('date', drop=True)
      industry_id_level2_k_data = industry_id_level2_k_data.groupby('industry_id_level2').apply(lambda x: x.set_index('date', drop=True))

      industry_id_level2_k_data = industry_id_level2_k_data.groupby(level=0).apply(lambda x: self.func(x, 'industry_id_level2_pctChg', 'industry_id_level2_close')).round(5)
      industry_id_level2_k_data['industry_id_level2_preclose'] = industry_id_level2_k_data['industry_id_level2_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0])).round(5)
      industry_id_level2_k_data['industry_id_level2_open'] = industry_id_level2_k_data['industry_id_level2_preclose']*(industry_id_level2_k_data['industry_id_level2_open_ratio'].apply(lambda x: x + 1)).round(5)
      industry_id_level2_k_data['industry_id_level2_high'] = industry_id_level2_k_data['industry_id_level2_preclose']*(industry_id_level2_k_data['industry_id_level2_high_ratio'].apply(lambda x: x + 1)).round(5)
      industry_id_level2_k_data['industry_id_level2_low'] = industry_id_level2_k_data['industry_id_level2_preclose']*(industry_id_level2_k_data['industry_id_level2_low_ratio'].apply(lambda x: x + 1)).round(5)

      # 写出指数点数
      industry_id_level2_k_data_out = industry_id_level2_k_data[['industry_id_level2_open', 'industry_id_level2_close', 'industry_id_level2_high', 'industry_id_level2_low']]
      industry_id_level2_k_data_out = industry_id_level2_k_data_out.reset_index(level=0, drop=False)
      industry_id_level2_k_data_out = industry_id_level2_k_data_out.reset_index(level=0, drop=False)
      industry_id_level2_k_data_out.to_csv('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level2_' + str(year) + '.csv', mode='w', header=True, index=False)
      del industry_id_level2_k_data_out
      gc.collect()

      industry_id_level2_k_data['industry_id_level2_open_ratio_7d_avg'] = industry_id_level2_k_data['industry_id_level2_open_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level2_k_data['industry_id_level2_close_ratio_7d_avg'] = industry_id_level2_k_data['industry_id_level2_close_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level2_k_data['industry_id_level2_high_ratio_7d_avg'] = industry_id_level2_k_data['industry_id_level2_high_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level2_k_data['industry_id_level2_low_ratio_7d_avg'] = industry_id_level2_k_data['industry_id_level2_low_ratio'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=7, center=False).mean()).round(5)
      industry_id_level2_k_data['industry_id_level2_turn'] = industry_id_level2_k_data["industry_id_level2_turn"].map(lambda x: float2Bucket(float(x) * 100, 1, 0, 50, 50))

      industry_id_level2_k_data['industry_id_level2_peTTM'] = industry_id_level2_k_data["industry_id_level2_peTTM"].map(lambda x: float2Bucket(float(x) + 200, 0.5, 0, 400, 200))
      industry_id_level2_k_data['industry_id_level2_pcfNcfTTM'] = industry_id_level2_k_data["industry_id_level2_pcfNcfTTM"].map(lambda x: float2Bucket(float(x) + 200, 0.5, 0, 400, 200))
      industry_id_level2_k_data['industry_id_level2_pbMRQ'] = industry_id_level2_k_data["industry_id_level2_pbMRQ"].map(lambda x: float2Bucket(float(x), 2, 0, 100, 200))
      industry_id_level2_k_data['industry_id_level2_market_value'] = industry_id_level2_k_data["industry_id_level2_market_value"].map(lambda x: None if x == '' else bignumber2Bucket(float(x), 1.25, 60))

      # kdj 5, 9, 19, 36, 45, 73，
      # 任意初始化，超过30天后的kdj值基本一样
      tmp = industry_id_level2_k_data[["industry_id_level2_low", "industry_id_level2_high", "industry_id_level2_close"]]
      for day_cnt in [5, 9, 19, 73]:
      # industry_id_level2_k_data['industry_id_level2_rsv'] = industry_id_level2_k_data[["industry_id_level2_low", "industry_id_level2_high", "industry_id_level2_close"]].groupby(level=0).apply(lambda x: (x.close-x.low.rolling(min_periods=1, window=day_cnt, center=False).min())/(x.high.rolling(min_periods=1, window=day_cnt, center=False).max()-x.low.rolling(min_periods=1, window=day_cnt, center=False).min()))
         tmp['min'] = tmp["industry_id_level2_low"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
         tmp['max'] = tmp["industry_id_level2_high"].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
         industry_id_level2_k_data['industry_id_level2_rsv_' + str(day_cnt)] = ((tmp["industry_id_level2_close"] - tmp['min'])/(tmp['max'] - tmp['min'])).round(5)
         industry_id_level2_k_data['industry_id_level2_k_value_' + str(day_cnt)] = industry_id_level2_k_data['industry_id_level2_rsv_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean()).round(5)
         industry_id_level2_k_data['industry_id_level2_d_value_' + str(day_cnt)] = industry_id_level2_k_data['industry_id_level2_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.ewm(alpha=1.0/3, adjust=False).mean()).round(5)
         industry_id_level2_k_data['industry_id_level2_j_value_' + str(day_cnt)] = (3 * industry_id_level2_k_data['industry_id_level2_k_value_' + str(day_cnt)] - 2 * industry_id_level2_k_data['industry_id_level2_d_value_' + str(day_cnt)]).round(5)
         # industry_id_level2_k_data['industry_id_level2_k_value_trend_' + str(day_cnt)] = industry_id_level2_k_data['industry_id_level2_k_value_' + str(day_cnt)].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: (y[1])-y[0]))
         # industry_id_level2_k_data['industry_id_level2_kd_value' + str(day_cnt)] = (industry_id_level2_k_data['industry_id_level2_k_value_' + str(day_cnt)] - industry_id_level2_k_data['industry_id_level2_d_value_' + str(day_cnt)]).rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1))

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
      industry_id_level2_k_data['industry_id_level2_macd_dif_ratio'] = ((tmp['macd_dif']-tmp['macd_dif_min'])/(tmp['macd_dif_max']-tmp['macd_dif_min'])).round(5)
      for day_cnt in [2, 3, 5, 10, 20, 40]:
         industry_id_level2_k_data['industry_id_level2_macd_dif_' + str(day_cnt)] = tmp['macd_dif'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         industry_id_level2_k_data['industry_id_level2_macd_dea_' + str(day_cnt)] = tmp['macd_dea'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         industry_id_level2_k_data['industry_id_level2_macd_' + str(day_cnt)] = tmp['macd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 0 if (y[day_cnt-1]-y[0])<0 else 1))
         industry_id_level2_k_data['industry_id_level2_macd_positive_ratio_' + str(day_cnt)] = industry_id_level2_k_data['industry_id_level2_macd_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean()).round(5)
      industry_id_level2_k_data['industry_id_level2_macd_dif_dea'] = (tmp['macd_dif']-tmp['macd_dea']).groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda x: 2 if x[1]>0 and x[0]<0 else (0 if x[1]<0 and x[0]>0 else 1)))

      # boll线
      for day_cnt in [2, 3, 5, 10, 20, 40]:
         tmp['mb_' + str(day_cnt)] = tmp['industry_id_level2_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
         tmp['md_' + str(day_cnt)] = tmp['industry_id_level2_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).std())
         tmp['up_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] + 2 * tmp['md_' + str(day_cnt)]
         tmp['dn_' + str(day_cnt)] = tmp['mb_' + str(day_cnt)] - 2 * tmp['md_' + str(day_cnt)]
         industry_id_level2_k_data['industry_id_level2_width_' + str(day_cnt)] = (4 * tmp['md_' + str(day_cnt)] / tmp['mb_' + str(day_cnt)]).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)
         industry_id_level2_k_data['industry_id_level2_close_mb' + str(day_cnt) + '_diff'] = ((tmp['industry_id_level2_close'] - tmp['mb_' + str(day_cnt)])/(2 * tmp['md_' + str(day_cnt)])).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)

      # cr指标
      tmp = industry_id_level2_k_data[["industry_id_level2_close", "industry_id_level2_open", "industry_id_level2_high", "industry_id_level2_low"]]
      tmp['cr_m'] = (tmp["industry_id_level2_close"] + tmp["industry_id_level2_open"] + tmp["industry_id_level2_high"] + tmp["industry_id_level2_low"])/4
      tmp['cr_ym'] = tmp['cr_m'].groupby(level=0).apply(lambda x: x.rolling(min_periods=2, window=2, center=False).apply(lambda y: y[0]))
      tmp['cr_p1_day'] = (tmp['industry_id_level2_high'] - tmp['cr_ym']).apply(lambda x: max(x, 0))
      tmp['cr_p2_day'] = (tmp['cr_ym'] - tmp['industry_id_level2_low']).apply(lambda x: max(x, 0))
      for day_cnt in [26]:
         tmp['cr_p1_' + str(day_cnt) + 'd'] = tmp['cr_p1_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         tmp['cr_p2_' + str(day_cnt) + 'd'] = tmp['cr_p2_day'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         tmp['cr_' + str(day_cnt) + 'd'] = tmp['cr_p1_' + str(day_cnt) + 'd'] / tmp['cr_p2_' + str(day_cnt) + 'd']
         tmp['cr_a_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=5, center=False).mean())
         tmp['cr_b_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=10, center=False).mean())
         tmp['cr_c_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=20, center=False).mean())
         tmp['cr_d_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=60, center=False).mean())
         industry_id_level2_k_data['industry_id_level2_cr_bias_' + str(day_cnt) + 'd'] = (tmp['cr_' + str(day_cnt) + 'd']/tmp['cr_a_' + str(day_cnt) + 'd']).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)
         industry_id_level2_k_data['industry_id_level2_cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: float2Bucket(float(x)*100, 0.1, 0, 300, 30))

         tmp['cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_a_' + str(day_cnt) + 'd'] = tmp['cr_a_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_b_' + str(day_cnt) + 'd'] = tmp['cr_b_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_c_' + str(day_cnt) + 'd'] = tmp['cr_c_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_d_' + str(day_cnt) + 'd'] = tmp['cr_d_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         # bucket空间可以设置成 75万，多个特征可以共享embedding
         industry_id_level2_k_data['industry_id_level2_cr_trend_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].str.cat([tmp['cr_a_' + str(day_cnt) + 'd'],tmp['cr_b_' + str(day_cnt) + 'd'],tmp['cr_c_' + str(day_cnt) + 'd'],tmp['cr_d_' + str(day_cnt) + 'd']], sep='_').apply(lambda x: self.tools.hash_bucket(x, 750000))
         for day_cnt_new in range(4):
            industry_id_level2_k_data['industry_id_level2_cr_trend_' + str(day_cnt) + 'd' + '_' + str(day_cnt_new)] = industry_id_level2_k_data['industry_id_level2_cr_trend_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt_new+2, window=day_cnt_new+2, center=False).apply(lambda y: y[0]))

      # rsi指标
      tmp = industry_id_level2_k_data[["industry_id_level2_close", "industry_id_level2_preclose"]]
      tmp['price_dif'] = tmp["industry_id_level2_close"] - tmp["industry_id_level2_preclose"]
      tmp['rsi_positive'] = tmp['price_dif'].apply(lambda x: max(x, 0))
      tmp['rsi_all'] = tmp['price_dif'].apply(lambda x: abs(x))
      for day_cnt in (3, 5, 10, 20, 40):
         tmp['rsi_positive_sum'] = tmp['rsi_positive'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         tmp['rsi_all_sum'] = tmp['rsi_all'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).sum())
         industry_id_level2_k_data['industry_id_level2_rsi_' + str(day_cnt) + 'd'] =  (tmp['rsi_positive_sum'] / tmp['rsi_all_sum']).round(5)

      tmp = industry_id_level2_k_data[["industry_id_level2_turn", "industry_id_level2_close"]]
      day_cnt_list = [3, 5, 10, 20, 30, 60, 120, 240]
      for index in range(len(day_cnt_list)):
         day_cnt = day_cnt_list[index]
         day_cnt_last = day_cnt_list[index-1]
         industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level2_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
         industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = tmp['industry_id_level2_turn']/industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg']
         industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg']/tmp['industry_id_level2_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
         industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'min_dif'] = tmp['industry_id_level2_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())/industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg']
         if index>0:
            industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + 'd' + '_avg']/industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg']
            industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + 'd' + 'max_dif']/industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'max_dif']
            industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + 'd' + 'min_dif']/industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'min_dif']

         tmp['industry_id_level2_close_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level2_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).mean())
         industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'avg_dif'] = tmp['industry_id_level2_close']/tmp['industry_id_level2_close_' + str(day_cnt) + 'd' + '_avg']
         industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'max_dif'] = tmp['industry_id_level2_close']/tmp['industry_id_level2_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).max())
         industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'min_dif'] = tmp['industry_id_level2_close']/tmp['industry_id_level2_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).min())
         industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + '_dif'] = tmp['industry_id_level2_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: 1.0*(y[day_cnt-1])/y[0]))
         if index>0:
            industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = tmp['industry_id_level2_close_' + str(day_cnt_last) + 'd' + '_avg']/tmp['industry_id_level2_close_' + str(day_cnt) + 'd' + '_avg']

      for index in range(len(day_cnt_list)):
         day_cnt = day_cnt_list[index]
         day_cnt_last = day_cnt_list[index - 1]
         if index > 0:
            industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 2, 100))
            industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 2, 100))
            industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 2, 100))
            industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt_last) + '_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: float2Bucket(float(x), 50, 0, 4, 200))
         industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + '_avg'].map(lambda x: min(2, round(x/100, 4)))
         industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
         industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: round(x, 4))
         industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level2_k_data['industry_id_level2_turn_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: round(x, 4))

         industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'avg_dif'] = industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'avg_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
         industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'max_dif'] = industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'max_dif'].map(lambda x: float2Bucket(float(x), 100, 0, 1, 100))
         industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'min_dif'] = industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + 'min_dif'].map(lambda x: float2Bucket(float(x), 50, 1, 10, 500))
         industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + '_dif'] = industry_id_level2_k_data['industry_id_level2_close_' + str(day_cnt) + 'd' + '_dif'].map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))

      # 还需要再做一些量价类特征，比如高位放量，低位缩量等等
      for day_cnt in [3, 5, 10, 20, 30, 60, 120, 240]:
         industry_id_level2_k_data['industry_id_level2_max_turn_index' + str(day_cnt) + 'd'] = tmp['industry_id_level2_turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmax()))
         industry_id_level2_k_data['industry_id_level2_max_close_index' + str(day_cnt) + 'd'] = tmp['industry_id_level2_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmax()))
         industry_id_level2_k_data['industry_id_level2_min_close_index' + str(day_cnt) + 'd'] = tmp['industry_id_level2_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=1, window=day_cnt, center=False).apply(lambda y: y.argmin()))
         industry_id_level2_k_data = industry_id_level2_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'industry_id_level2_max_turn_index' + str(day_cnt) + 'd', 'industry_id_level2_close', 'industry_id_level2_max_turn_close' + str(day_cnt) + 'd', day_cnt))
         industry_id_level2_k_data = industry_id_level2_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'industry_id_level2_max_close_index' + str(day_cnt) + 'd', 'industry_id_level2_turn', 'industry_id_level2_max_closs_turn' + str(day_cnt) + 'd', day_cnt))
         industry_id_level2_k_data = industry_id_level2_k_data.groupby(level=0).apply(lambda x: self._rolling(x, 'industry_id_level2_min_close_index' + str(day_cnt) + 'd', 'industry_id_level2_turn', 'industry_id_level2_min_closs_turn' + str(day_cnt) + 'd', day_cnt))

         industry_id_level2_k_data['industry_id_level2_max_turn_close' + str(day_cnt) + 'd'] = (tmp['industry_id_level2_close']/industry_id_level2_k_data['industry_id_level2_max_turn_close' + str(day_cnt) + 'd']).map(lambda x: float2Bucket(float(x), 50, 0, 10, 500))
         industry_id_level2_k_data['industry_id_level2_max_closs_turn' + str(day_cnt) + 'd'] = (industry_id_level2_k_data['industry_id_level2_max_closs_turn' + str(day_cnt) + 'd']/tmp['industry_id_level2_turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))
         industry_id_level2_k_data['industry_id_level2_min_closs_turn' + str(day_cnt) + 'd'] = (industry_id_level2_k_data['industry_id_level2_min_closs_turn' + str(day_cnt) + 'd']/tmp['industry_id_level2_turn']).map(lambda x: float2Bucket(float(x), 10, 0, 20, 200))

      industry_id_level2_k_data = industry_id_level2_k_data.reset_index(level=0, drop=True)
      industry_id_level2_k_data = industry_id_level2_k_data.reset_index(level=0, drop=False)
      selected_columns = ['date','industry_id_level2','industry_id_level2_open_ratio','industry_id_level2_close_ratio','industry_id_level2_high_ratio','industry_id_level2_low_ratio','industry_id_level2_turn','industry_id_level2_pctChg','industry_id_level2_peTTM','industry_id_level2_pcfNcfTTM','industry_id_level2_pbMRQ','industry_id_level2_rise_ratio','industry_id_level2_market_value','industry_id_level2_open_ratio_7d_avg','industry_id_level2_close_ratio_7d_avg','industry_id_level2_high_ratio_7d_avg','industry_id_level2_low_ratio_7d_avg','industry_id_level2_rsv_5','industry_id_level2_k_value_5','industry_id_level2_d_value_5','industry_id_level2_j_value_5','industry_id_level2_rsv_9','industry_id_level2_k_value_9','industry_id_level2_d_value_9','industry_id_level2_j_value_9','industry_id_level2_rsv_19','industry_id_level2_k_value_19','industry_id_level2_d_value_19','industry_id_level2_j_value_19','industry_id_level2_rsv_73','industry_id_level2_k_value_73','industry_id_level2_d_value_73','industry_id_level2_j_value_73','industry_id_level2_macd_positive','industry_id_level2_macd_dif_ratio','industry_id_level2_macd_dif_2','industry_id_level2_macd_dea_2','industry_id_level2_macd_2','industry_id_level2_macd_positive_ratio_2','industry_id_level2_macd_dif_3','industry_id_level2_macd_dea_3','industry_id_level2_macd_3','industry_id_level2_macd_positive_ratio_3','industry_id_level2_macd_dif_5','industry_id_level2_macd_dea_5','industry_id_level2_macd_5','industry_id_level2_macd_positive_ratio_5','industry_id_level2_macd_dif_10','industry_id_level2_macd_dea_10','industry_id_level2_macd_10','industry_id_level2_macd_positive_ratio_10','industry_id_level2_macd_dif_20','industry_id_level2_macd_dea_20','industry_id_level2_macd_20','industry_id_level2_macd_positive_ratio_20','industry_id_level2_macd_dif_40','industry_id_level2_macd_dea_40','industry_id_level2_macd_40','industry_id_level2_macd_positive_ratio_40','industry_id_level2_macd_dif_dea','industry_id_level2_width_2','industry_id_level2_close_mb2_diff','industry_id_level2_width_3','industry_id_level2_close_mb3_diff','industry_id_level2_width_5','industry_id_level2_close_mb5_diff','industry_id_level2_width_10','industry_id_level2_close_mb10_diff','industry_id_level2_width_20','industry_id_level2_close_mb20_diff','industry_id_level2_width_40','industry_id_level2_close_mb40_diff','industry_id_level2_cr_bias_26d','industry_id_level2_cr_26d','industry_id_level2_cr_trend_26d','industry_id_level2_cr_trend_26d_0','industry_id_level2_cr_trend_26d_1','industry_id_level2_cr_trend_26d_2','industry_id_level2_cr_trend_26d_3','industry_id_level2_rsi_3d','industry_id_level2_rsi_5d','industry_id_level2_rsi_10d','industry_id_level2_rsi_20d','industry_id_level2_rsi_40d','industry_id_level2_turn_3d_avg','industry_id_level2_turn_3davg_dif','industry_id_level2_turn_3dmax_dif','industry_id_level2_turn_3dmin_dif','industry_id_level2_close_3davg_dif','industry_id_level2_close_3dmax_dif','industry_id_level2_close_3dmin_dif','industry_id_level2_close_3d_dif','industry_id_level2_turn_5d_avg','industry_id_level2_turn_5davg_dif','industry_id_level2_turn_5dmax_dif','industry_id_level2_turn_5dmin_dif','industry_id_level2_turn_3_5d_avg','industry_id_level2_turn_3_5dmax_dif','industry_id_level2_turn_3_5dmin_dif','industry_id_level2_close_5davg_dif','industry_id_level2_close_5dmax_dif','industry_id_level2_close_5dmin_dif','industry_id_level2_close_5d_dif','industry_id_level2_close_3_5d_avg','industry_id_level2_turn_10d_avg','industry_id_level2_turn_10davg_dif','industry_id_level2_turn_10dmax_dif','industry_id_level2_turn_10dmin_dif','industry_id_level2_turn_5_10d_avg','industry_id_level2_turn_5_10dmax_dif','industry_id_level2_turn_5_10dmin_dif','industry_id_level2_close_10davg_dif','industry_id_level2_close_10dmax_dif','industry_id_level2_close_10dmin_dif','industry_id_level2_close_10d_dif','industry_id_level2_close_5_10d_avg','industry_id_level2_turn_20d_avg','industry_id_level2_turn_20davg_dif','industry_id_level2_turn_20dmax_dif','industry_id_level2_turn_20dmin_dif','industry_id_level2_turn_10_20d_avg','industry_id_level2_turn_10_20dmax_dif','industry_id_level2_turn_10_20dmin_dif','industry_id_level2_close_20davg_dif','industry_id_level2_close_20dmax_dif','industry_id_level2_close_20dmin_dif','industry_id_level2_close_20d_dif','industry_id_level2_close_10_20d_avg','industry_id_level2_turn_30d_avg','industry_id_level2_turn_30davg_dif','industry_id_level2_turn_30dmax_dif','industry_id_level2_turn_30dmin_dif','industry_id_level2_turn_20_30d_avg','industry_id_level2_turn_20_30dmax_dif','industry_id_level2_turn_20_30dmin_dif','industry_id_level2_close_30davg_dif','industry_id_level2_close_30dmax_dif','industry_id_level2_close_30dmin_dif','industry_id_level2_close_30d_dif','industry_id_level2_close_20_30d_avg','industry_id_level2_turn_60d_avg','industry_id_level2_turn_60davg_dif','industry_id_level2_turn_60dmax_dif','industry_id_level2_turn_60dmin_dif','industry_id_level2_turn_30_60d_avg','industry_id_level2_turn_30_60dmax_dif','industry_id_level2_turn_30_60dmin_dif','industry_id_level2_close_60davg_dif','industry_id_level2_close_60dmax_dif','industry_id_level2_close_60dmin_dif','industry_id_level2_close_60d_dif','industry_id_level2_close_30_60d_avg','industry_id_level2_turn_120d_avg','industry_id_level2_turn_120davg_dif','industry_id_level2_turn_120dmax_dif','industry_id_level2_turn_120dmin_dif','industry_id_level2_turn_60_120d_avg','industry_id_level2_turn_60_120dmax_dif','industry_id_level2_turn_60_120dmin_dif','industry_id_level2_close_120davg_dif','industry_id_level2_close_120dmax_dif','industry_id_level2_close_120dmin_dif','industry_id_level2_close_120d_dif','industry_id_level2_close_60_120d_avg','industry_id_level2_turn_240d_avg','industry_id_level2_turn_240davg_dif','industry_id_level2_turn_240dmax_dif','industry_id_level2_turn_240dmin_dif','industry_id_level2_turn_120_240d_avg','industry_id_level2_turn_120_240dmax_dif','industry_id_level2_turn_120_240dmin_dif','industry_id_level2_close_240davg_dif','industry_id_level2_close_240dmax_dif','industry_id_level2_close_240dmin_dif','industry_id_level2_close_240d_dif','industry_id_level2_close_120_240d_avg','industry_id_level2_max_turn_index3d','industry_id_level2_max_close_index3d','industry_id_level2_min_close_index3d','industry_id_level2_max_turn_close3d','industry_id_level2_max_closs_turn3d','industry_id_level2_min_closs_turn3d','industry_id_level2_max_turn_index5d','industry_id_level2_max_close_index5d','industry_id_level2_min_close_index5d','industry_id_level2_max_turn_close5d','industry_id_level2_max_closs_turn5d','industry_id_level2_min_closs_turn5d','industry_id_level2_max_turn_index10d','industry_id_level2_max_close_index10d','industry_id_level2_min_close_index10d','industry_id_level2_max_turn_close10d','industry_id_level2_max_closs_turn10d','industry_id_level2_min_closs_turn10d','industry_id_level2_max_turn_index20d','industry_id_level2_max_close_index20d','industry_id_level2_min_close_index20d','industry_id_level2_max_turn_close20d','industry_id_level2_max_closs_turn20d','industry_id_level2_min_closs_turn20d','industry_id_level2_max_turn_index30d','industry_id_level2_max_close_index30d','industry_id_level2_min_close_index30d','industry_id_level2_max_turn_close30d','industry_id_level2_max_closs_turn30d','industry_id_level2_min_closs_turn30d','industry_id_level2_max_turn_index60d','industry_id_level2_max_close_index60d','industry_id_level2_min_close_index60d','industry_id_level2_max_turn_close60d','industry_id_level2_max_closs_turn60d','industry_id_level2_min_closs_turn60d','industry_id_level2_max_turn_index120d','industry_id_level2_max_close_index120d','industry_id_level2_min_close_index120d','industry_id_level2_max_turn_close120d','industry_id_level2_max_closs_turn120d','industry_id_level2_min_closs_turn120d','industry_id_level2_max_turn_index240d','industry_id_level2_max_close_index240d','industry_id_level2_min_close_index240d','industry_id_level2_max_turn_close240d','industry_id_level2_max_closs_turn240d','industry_id_level2_min_closs_turn240d']
      industry_id_level2_k_data = industry_id_level2_k_data[selected_columns]

      feature_all = pd.merge(feature_all, industry_id_level2_k_data, how="left", left_on=["date",'industry_id_level2'],right_on=["date",'industry_id_level2'])
      feature_all = feature_all.sort_values(['date', 'code'])
      del industry_id_level2_k_data
      del tmp
      gc.collect()

      # 三级行业特征
      raw_k_data = pd.read_csv(self.k_file_path)
      raw_k_data_his = pd.read_csv(self.k_file_path_his)
      if is_predict:
         raw_k_data_his = raw_k_data_his[raw_k_data_his['date']>self.tools.get_recent_month_date(self.date_start, -14)]
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
          industry_id_level3_k_data_his = industry_id_level3_k_data_his[['industry_id_level3','date','industry_id_level3_close']]
          industry_id_level3_k_data_his['date'] = pd.to_datetime(industry_id_level3_k_data_his['date'])
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
      industry_id_level3_k_data_out = industry_id_level3_k_data[['industry_id_level3_open', 'industry_id_level3_close', 'industry_id_level3_high', 'industry_id_level3_low']]
      industry_id_level3_k_data_out = industry_id_level3_k_data_out.reset_index(level=0, drop=False)
      industry_id_level3_k_data_out = industry_id_level3_k_data_out.reset_index(level=0, drop=False)
      industry_id_level3_k_data_out.to_csv('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level3_' + str(year) + '.csv', mode='w', header=True, index=False)
      del industry_id_level3_k_data_out
      gc.collect()

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
         industry_id_level3_k_data['industry_id_level3_width_' + str(day_cnt)] = (4 * tmp['md_' + str(day_cnt)] / tmp['mb_' + str(day_cnt)]).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)
         industry_id_level3_k_data['industry_id_level3_close_mb' + str(day_cnt) + '_diff'] = ((tmp['industry_id_level3_close'] - tmp['mb_' + str(day_cnt)])/(2 * tmp['md_' + str(day_cnt)])).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)

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
         industry_id_level3_k_data['industry_id_level3_cr_bias_' + str(day_cnt) + 'd'] = (tmp['cr_' + str(day_cnt) + 'd']/tmp['cr_a_' + str(day_cnt) + 'd']).apply(lambda x: max(x, -3) if x<0 else min(x, 3)).round(5)
         industry_id_level3_k_data['industry_id_level3_cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: float2Bucket(float(x)*100, 0.1, 0, 300, 30))

         tmp['cr_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_a_' + str(day_cnt) + 'd'] = tmp['cr_a_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_b_' + str(day_cnt) + 'd'] = tmp['cr_b_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_c_' + str(day_cnt) + 'd'] = tmp['cr_c_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         tmp['cr_d_' + str(day_cnt) + 'd'] = tmp['cr_d_' + str(day_cnt) + 'd'].map(lambda x: str(float2Bucket(float(x)*100, 0.05, 0, 300, 15)))
         # bucket空间可以设置成 75万，多个特征可以共享embedding
         industry_id_level3_k_data['industry_id_level3_cr_trend_' + str(day_cnt) + 'd'] = tmp['cr_' + str(day_cnt) + 'd'].str.cat([tmp['cr_a_' + str(day_cnt) + 'd'],tmp['cr_b_' + str(day_cnt) + 'd'],tmp['cr_c_' + str(day_cnt) + 'd'],tmp['cr_d_' + str(day_cnt) + 'd']], sep='_').apply(lambda x: self.tools.hash_bucket(x, 750000))
         for day_cnt_new in range(4):
            industry_id_level3_k_data['industry_id_level3_cr_trend_' + str(day_cnt) + 'd' + '_' + str(day_cnt_new)] = industry_id_level3_k_data['industry_id_level3_cr_trend_' + str(day_cnt) + 'd'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt_new+2, window=day_cnt_new+2, center=False).apply(lambda y: y[0]))

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
      del tmp
      gc.collect()

      # 大盘趋势类特征
      raw_k_data = pd.read_csv(self.k_file_path)
      raw_k_data_his = pd.read_csv(self.k_file_path_his)
      if is_predict:
         raw_k_data_his = raw_k_data_his[raw_k_data_his['date']>self.tools.get_recent_month_date(self.date_start, -14)]
      raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
      raw_k_data = raw_k_data[raw_k_data['industry_id_level3'] > 0]
      del raw_k_data_his
      gc.collect()
      raw_k_data["tradestatus"] = pd.to_numeric(raw_k_data["tradestatus"], errors='coerce')
      raw_k_data["turn"] = pd.to_numeric(raw_k_data["turn"], errors='coerce')
      raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')
      raw_k_data = raw_k_data[(raw_k_data['tradestatus'] == 1) & (raw_k_data['turn'] > 0) & (raw_k_data['pctChg'] <= 20) & (raw_k_data['pctChg'] >= -20)]
      raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))
      raw_k_data['is_new'] = raw_k_data["pctChg"].groupby(level=0).apply(lambda x: x.rolling(min_periods=20, window=20, center=False).apply(lambda y: y[0]))
      raw_k_data = raw_k_data[raw_k_data['is_new'].map(lambda x: False if np.isnan(x) else True)]
      raw_k_data = raw_k_data.reset_index(level=0, drop=True)
      raw_k_data = raw_k_data.reset_index(level=0, drop=False)

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

      all_data_raw = raw_k_data[['date', 'pctChg', 'turn', 'rise_ratio']].groupby('date').mean().round(5)
      all_data_raw.columns = ['all_pctChg', 'all_turn', 'all_rise_ratio']

      del raw_k_data
      gc.collect()

      day_cnt_list = [3, 5, 10]
      for index in range(len(day_cnt_list)):
         day_cnt = day_cnt_list[index]
         all_data_raw['all_pctChg_' + str(day_cnt)] = all_data_raw['all_pctChg'].rolling(min_periods=1, window=day_cnt, center=False).mean().round(5)
         all_data_raw['all_turn_' + str(day_cnt)] = all_data_raw['all_turn'].rolling(min_periods=1, window=day_cnt, center=False).mean()
         all_data_raw['all_rise_ratio_' + str(day_cnt)] = all_data_raw['all_rise_ratio'].rolling(min_periods=1, window=day_cnt, center=False).mean().round(5)
      all_data_raw['all_turn'] = all_data_raw['all_turn'].map(lambda x: float2Bucket(float(x), 2, 0, 25, 50))

      for index in range(len(day_cnt_list)):
         day_cnt = day_cnt_list[index]
         all_data_raw['all_turn_' + str(day_cnt)] = all_data_raw['all_turn_' + str(day_cnt)].map(lambda x: float2Bucket(float(x), 2, 0, 25, 50))

      all_data_raw = all_data_raw.reset_index(level=0, drop=False)
      zs_data = feature_all[(feature_all['code'] == 'sh.000001') | (feature_all['code'] == 'sz.399001') | (feature_all['code'] == 'sz.399006')]
      zs_data['code_market'] = zs_data['code'].map(lambda x: self.tools.zh_code_market(x))
      zs_data = zs_data[['date','code_market','open_ratio','close_ratio','high_ratio','low_ratio','pctChg','open_ratio_7d_avg','close_ratio_7d_avg','high_ratio_7d_avg','low_ratio_7d_avg','turn','rsv_3','k_value_3','d_value_3','j_value_3','rsv_5','k_value_5','d_value_5','j_value_5','rsv_9','k_value_9','d_value_9','j_value_9','rsv_19','k_value_19','d_value_19','j_value_19','rsv_73','k_value_73','d_value_73','j_value_73','macd_positive','macd_dif_ratio','macd_dif_2','macd_dea_2','macd_2','macd_positive_ratio_2','macd_dif_3','macd_dea_3','macd_3','macd_positive_ratio_3','macd_dif_5','macd_dea_5','macd_5','macd_positive_ratio_5','macd_dif_10','macd_dea_10','macd_10','macd_positive_ratio_10','macd_dif_20','macd_dea_20','macd_20','macd_positive_ratio_20','macd_dif_40','macd_dea_40','macd_40','macd_positive_ratio_40','macd_dif_dea','width_2','close_mb2_diff','width_3','close_mb3_diff','width_5','close_mb5_diff','width_10','close_mb10_diff','width_20','close_mb20_diff','width_40','close_mb40_diff','cr_bias_26d','cr_26d','cr_trend_26d','cr_trend_26d_0','cr_trend_26d_1','cr_trend_26d_2','cr_trend_26d_3','rsi_3d','rsi_5d','rsi_10d','rsi_20d','rsi_40d','turn_3d_avg','turn_3davg_dif','turn_3dmax_dif','turn_3dmin_dif','close_3davg_dif','close_3dmax_dif','close_3dmin_dif','close_3d_dif','turn_5d_avg','turn_5davg_dif','turn_5dmax_dif','turn_5dmin_dif','turn_3_5d_avg','turn_3_5dmax_dif','turn_3_5dmin_dif','close_5davg_dif','close_5dmax_dif','close_5dmin_dif','close_5d_dif','close_3_5d_avg','turn_10d_avg','turn_10davg_dif','turn_10dmax_dif','turn_10dmin_dif','turn_5_10d_avg','turn_5_10dmax_dif','turn_5_10dmin_dif','close_10davg_dif','close_10dmax_dif','close_10dmin_dif','close_10d_dif','close_5_10d_avg','turn_20d_avg','turn_20davg_dif','turn_20dmax_dif','turn_20dmin_dif','turn_10_20d_avg','turn_10_20dmax_dif','turn_10_20dmin_dif','close_20davg_dif','close_20dmax_dif','close_20dmin_dif','close_20d_dif','close_10_20d_avg','turn_30d_avg','turn_30davg_dif','turn_30dmax_dif','turn_30dmin_dif','turn_20_30d_avg','turn_20_30dmax_dif','turn_20_30dmin_dif','close_30davg_dif','close_30dmax_dif','close_30dmin_dif','close_30d_dif','close_20_30d_avg','turn_60d_avg','turn_60davg_dif','turn_60dmax_dif','turn_60dmin_dif','turn_30_60d_avg','turn_30_60dmax_dif','turn_30_60dmin_dif','close_60davg_dif','close_60dmax_dif','close_60dmin_dif','close_60d_dif','close_30_60d_avg','turn_120d_avg','turn_120davg_dif','turn_120dmax_dif','turn_120dmin_dif','turn_60_120d_avg','turn_60_120dmax_dif','turn_60_120dmin_dif','close_120davg_dif','close_120dmax_dif','close_120dmin_dif','close_120d_dif','close_60_120d_avg','turn_240d_avg','turn_240davg_dif','turn_240dmax_dif','turn_240dmin_dif','turn_120_240d_avg','turn_120_240dmax_dif','turn_120_240dmin_dif','close_240davg_dif','close_240dmax_dif','close_240dmin_dif','close_240d_dif','close_120_240d_avg','max_turn_index3d','max_close_index3d','min_close_index3d','max_turn_close3d','max_closs_turn3d','min_closs_turn3d','max_turn_index5d','max_close_index5d','min_close_index5d','max_turn_close5d','max_closs_turn5d','min_closs_turn5d','max_turn_index10d','max_close_index10d','min_close_index10d','max_turn_close10d','max_closs_turn10d','min_closs_turn10d','max_turn_index20d','max_close_index20d','min_close_index20d','max_turn_close20d','max_closs_turn20d','min_closs_turn20d','max_turn_index30d','max_close_index30d','min_close_index30d','max_turn_close30d','max_closs_turn30d','min_closs_turn30d','max_turn_index60d','max_close_index60d','min_close_index60d','max_turn_close60d','max_closs_turn60d','min_closs_turn60d','max_turn_index120d','max_close_index120d','min_close_index120d','max_turn_close120d','max_closs_turn120d','min_closs_turn120d','max_turn_index240d','max_close_index240d','min_close_index240d','max_turn_close240d','max_closs_turn240d','min_closs_turn240d']]
      zs_data.columns = ['date','code_market','zs_open_ratio','zs_close_ratio','zs_high_ratio','zs_low_ratio','zs_pctChg','zs_open_ratio_7d_avg','zs_close_ratio_7d_avg','zs_high_ratio_7d_avg','zs_low_ratio_7d_avg','zs_turn','zs_rsv_3','zs_k_value_3','zs_d_value_3','zs_j_value_3','zs_rsv_5','zs_k_value_5','zs_d_value_5','zs_j_value_5','zs_rsv_9','zs_k_value_9','zs_d_value_9','zs_j_value_9','zs_rsv_19','zs_k_value_19','zs_d_value_19','zs_j_value_19','zs_rsv_73','zs_k_value_73','zs_d_value_73','zs_j_value_73','zs_macd_positive','zs_macd_dif_ratio','zs_macd_dif_2','zs_macd_dea_2','zs_macd_2','zs_macd_positive_ratio_2','zs_macd_dif_3','zs_macd_dea_3','zs_macd_3','zs_macd_positive_ratio_3','zs_macd_dif_5','zs_macd_dea_5','zs_macd_5','zs_macd_positive_ratio_5','zs_macd_dif_10','zs_macd_dea_10','zs_macd_10','zs_macd_positive_ratio_10','zs_macd_dif_20','zs_macd_dea_20','zs_macd_20','zs_macd_positive_ratio_20','zs_macd_dif_40','zs_macd_dea_40','zs_macd_40','zs_macd_positive_ratio_40','zs_macd_dif_dea','zs_width_2','zs_close_mb2_diff','zs_width_3','zs_close_mb3_diff','zs_width_5','zs_close_mb5_diff','zs_width_10','zs_close_mb10_diff','zs_width_20','zs_close_mb20_diff','zs_width_40','zs_close_mb40_diff','zs_cr_bias_26d','zs_cr_26d','zs_cr_trend_26d','zs_cr_trend_26d_0','zs_cr_trend_26d_1','zs_cr_trend_26d_2','zs_cr_trend_26d_3','zs_rsi_3d','zs_rsi_5d','zs_rsi_10d','zs_rsi_20d','zs_rsi_40d','zs_turn_3d_avg','zs_turn_3davg_dif','zs_turn_3dmax_dif','zs_turn_3dmin_dif','zs_close_3davg_dif','zs_close_3dmax_dif','zs_close_3dmin_dif','zs_close_3d_dif','zs_turn_5d_avg','zs_turn_5davg_dif','zs_turn_5dmax_dif','zs_turn_5dmin_dif','zs_turn_3_5d_avg','zs_turn_3_5dmax_dif','zs_turn_3_5dmin_dif','zs_close_5davg_dif','zs_close_5dmax_dif','zs_close_5dmin_dif','zs_close_5d_dif','zs_close_3_5d_avg','zs_turn_10d_avg','zs_turn_10davg_dif','zs_turn_10dmax_dif','zs_turn_10dmin_dif','zs_turn_5_10d_avg','zs_turn_5_10dmax_dif','zs_turn_5_10dmin_dif','zs_close_10davg_dif','zs_close_10dmax_dif','zs_close_10dmin_dif','zs_close_10d_dif','zs_close_5_10d_avg','zs_turn_20d_avg','zs_turn_20davg_dif','zs_turn_20dmax_dif','zs_turn_20dmin_dif','zs_turn_10_20d_avg','zs_turn_10_20dmax_dif','zs_turn_10_20dmin_dif','zs_close_20davg_dif','zs_close_20dmax_dif','zs_close_20dmin_dif','zs_close_20d_dif','zs_close_10_20d_avg','zs_turn_30d_avg','zs_turn_30davg_dif','zs_turn_30dmax_dif','zs_turn_30dmin_dif','zs_turn_20_30d_avg','zs_turn_20_30dmax_dif','zs_turn_20_30dmin_dif','zs_close_30davg_dif','zs_close_30dmax_dif','zs_close_30dmin_dif','zs_close_30d_dif','zs_close_20_30d_avg','zs_turn_60d_avg','zs_turn_60davg_dif','zs_turn_60dmax_dif','zs_turn_60dmin_dif','zs_turn_30_60d_avg','zs_turn_30_60dmax_dif','zs_turn_30_60dmin_dif','zs_close_60davg_dif','zs_close_60dmax_dif','zs_close_60dmin_dif','zs_close_60d_dif','zs_close_30_60d_avg','zs_turn_120d_avg','zs_turn_120davg_dif','zs_turn_120dmax_dif','zs_turn_120dmin_dif','zs_turn_60_120d_avg','zs_turn_60_120dmax_dif','zs_turn_60_120dmin_dif','zs_close_120davg_dif','zs_close_120dmax_dif','zs_close_120dmin_dif','zs_close_120d_dif','zs_close_60_120d_avg','zs_turn_240d_avg','zs_turn_240davg_dif','zs_turn_240dmax_dif','zs_turn_240dmin_dif','zs_turn_120_240d_avg','zs_turn_120_240dmax_dif','zs_turn_120_240dmin_dif','zs_close_240davg_dif','zs_close_240dmax_dif','zs_close_240dmin_dif','zs_close_240d_dif','zs_close_120_240d_avg','zs_max_turn_index3d','zs_max_close_index3d','zs_min_close_index3d','zs_max_turn_close3d','zs_max_closs_turn3d','zs_min_closs_turn3d','zs_max_turn_index5d','zs_max_close_index5d','zs_min_close_index5d','zs_max_turn_close5d','zs_max_closs_turn5d','zs_min_closs_turn5d','zs_max_turn_index10d','zs_max_close_index10d','zs_min_close_index10d','zs_max_turn_close10d','zs_max_closs_turn10d','zs_min_closs_turn10d','zs_max_turn_index20d','zs_max_close_index20d','zs_min_close_index20d','zs_max_turn_close20d','zs_max_closs_turn20d','zs_min_closs_turn20d','zs_max_turn_index30d','zs_max_close_index30d','zs_min_close_index30d','zs_max_turn_close30d','zs_max_closs_turn30d','zs_min_closs_turn30d','zs_max_turn_index60d','zs_max_close_index60d','zs_min_close_index60d','zs_max_turn_close60d','zs_max_closs_turn60d','zs_min_closs_turn60d','zs_max_turn_index120d','zs_max_close_index120d','zs_min_close_index120d','zs_max_turn_close120d','zs_max_closs_turn120d','zs_min_closs_turn120d','zs_max_turn_index240d','zs_max_close_index240d','zs_min_close_index240d','zs_max_turn_close240d','zs_max_closs_turn240d','zs_min_closs_turn240d',]
      all_data_raw = pd.merge(all_data_raw, zs_data, how="left", left_on=["date"],right_on=["date"])
      feature_all = pd.merge(feature_all, all_data_raw, how="left", left_on=["date",'code_market'],right_on=["date",'code_market'])
      del all_data_raw
      del zs_data
      gc.collect()
      return feature_all
if __name__ == '__main__':
   years = [2022]
   is_predict = True
   date_start = '2022-11-26'
   date_end = '2022-12-16'
   # years = [2008, 2009]
   # time.sleep(18000)
   for year in years:
      path = 'E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_'
      quater_path = 'E:/pythonProject/future/data/datafile/code_quarter_data_v2_all.csv'
      output_path = 'E:/pythonProject/future/data/datafile/feature/{year}_feature_v7.csv'.format(year=str(year))
      # raw_k_data = pd.read_csv(path + str(year) + '.csv')
      # raw_k_data.to_csv('E:/pythonProject/future/data/datafile/raw_feature/test_code_k_data_v5_' + str(year) + '.csv', mode='a', header=True, index=False)
      feature = Feature(path, year, quater_path, is_predict, date_start, date_end)
      feature_all = feature.feature_process()
      if os.path.isfile(output_path):
         feature_all.to_csv(output_path, mode='a', header=False, index=False)
      else:
         feature_all.to_csv(output_path, mode='w', header=True, index=False)
      del feature_all
      gc.collect()