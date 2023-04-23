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
   def __init__(self,  k_file_path, year, quarter_file_path, is_predict, date_start, date_end, model_name):
      self.k_file_path = k_file_path + str(year) + '.csv'
      self.k_file_path_his = k_file_path + str(year - 1) + '.csv'
      self.quarter_file_path = quarter_file_path
      self.year = year
      self.tools = Tools()
      self.is_predict = is_predict
      self.date_start = date_start
      self.date_end = date_end
      self.model_name = model_name

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
      raw_k_data = raw_k_data[(raw_k_data['industry_id_level3'] > 0) | (raw_k_data['code'] == 'sh.000001') | (raw_k_data['code'] == 'sz.399001') | (raw_k_data['code'] == 'sz.399006')]
      raw_k_data_his = pd.read_csv(self.k_file_path_his)
      raw_k_data_his = raw_k_data_his[(raw_k_data_his['industry_id_level3'] > 0) | (raw_k_data_his['code'] == 'sh.000001') | (raw_k_data_his['code'] == 'sz.399001') | (raw_k_data_his['code'] == 'sz.399006')]
      if is_predict:
         raw_k_data_his = raw_k_data_his[raw_k_data_his['date']>self.tools.get_recent_month_date(self.date_start, -14)]
      raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
      del raw_k_data_his
      gc.collect()
      raw_k_data["tradestatus"] = pd.to_numeric(raw_k_data["tradestatus"], errors='coerce')
      raw_k_data["turn"] = pd.to_numeric(raw_k_data["turn"], errors='coerce')
      raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')

      raw_k_data = raw_k_data[(raw_k_data['tradestatus'] == 1) & (raw_k_data['turn'] > 0) & (raw_k_data['pctChg'] < 21) & (raw_k_data['pctChg'] > -21)]

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

      feature_all = copy.deepcopy(raw_k_data[['industry_name_level1','industry_name_level2','industry_name_level3','industry_id_level1','industry_id_level2','industry_id_level3','code_market']])

      # 新增股票当日交易量放大的股票。当日换手率/最近10天（不包括当日）的最大换手率或者平均换手率，这两个特征需要交叉
      tmp = raw_k_data[['turn','close','amount']]
      tmp['amount'] = tmp['amount'] * 0.0000005
      # 当日换手率/最近10天（不包括当日）的最大换手率或者平均换手率
      # 当日收盘价格/最近10天（不包括当日）的最大收盘价或者平均收盘价格
      tmp['max_turn'] = tmp['turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=10, window=10, center=False).apply(lambda y: max(y[0:9])))
      tmp['min_close'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=10, window=10, center=False).apply(lambda y: min(y[0:9])))
      tmp['first_close'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=10, window=10, center=False).apply(lambda y: y[0]))
      tmp['avg_amount'] = tmp['amount'].groupby(level=0).apply(lambda x: x.rolling(min_periods=10, window=10, center=False).apply(lambda y: np.mean(y[0:9])))

      feature_all['max_turn_diff'] = (tmp['turn']/tmp['max_turn']).map(lambda x: float2Bucket(float(x), 1, 0, 10, 10))
      feature_all['avg_turn_diff'] = (tmp['turn']/tmp['avg_turn']).map(lambda x: float2Bucket(float(x), 1, 0, 20, 20))
      feature_all['min_close_diff'] = (tmp['close']/tmp['min_close']).map(lambda x: float2Bucket(float(x), 20, 0, 2, 40))
      feature_all['first_close_diff'] = (tmp['close']/tmp['first_close']).map(lambda x: float2Bucket(float(x), 20, 0, 2, 40))
      feature_all['avg_amount_diff'] = (tmp['amount']-tmp['avg_amount']).map(lambda x: float2Bucket(float(x), 1, 0, 100, 100))

      # bucket size分别为最大1400，2400
      feature_all['max_turn_min_close_cross'] = feature_all[['max_turn_diff','min_close_diff']].apply(lambda x: (x.max_turn_diff+1)*100 + x.min_close_diff, axis=1)
      feature_all['avg_turn_min_close_cross'] = feature_all[['avg_turn_diff','min_close_diff']].apply(lambda x: (x.avg_turn_diff+1)*100 + x.min_close_diff, axis=1)
      feature_all['avg_turn_first_close_cross'] = feature_all[['avg_turn_diff','first_close_diff']].apply(lambda x: (x.avg_turn_diff+1)*100 + x.first_close_diff, axis=1)

      # 新增股票的历史最高价和时间

      # 新增过去n天涨幅超过大盘，行业指数m个点的天数
      # 新增板块的最近n天的涨停股票数量，主要评估板块热度
      feature_all = feature_all.reset_index(level=0, drop=False)
      feature_all = feature_all.reset_index(level=0, drop=False)
      feature_all = feature_all.sort_values(['date', 'code'])
      feature_all = feature_all[feature_all['date'] > str(self.year)]
      if is_predict:
         feature_all = feature_all[(feature_all['date'] >= self.date_start) & (feature_all['date'] <= self.date_end)]
      del tmp
      del raw_k_data

      gc.collect()
      if self.is_predict:
         sample = feature_all
         sample['date'] = sample['date'].map(lambda x: int(str(x).replace('-', '')[:6]))
         sample = sample[sample['code_market'] != 0]
         sample['label_7'] = 0
         sample['label_7_real'] = 0
         sample['label_7_weight'] = 0
         sample['label_7_raw'] = 0
         sample['label_7_raw_real'] = 0
         sample['label_7_raw_weight'] = 0
         sample['label_15'] = 0
         sample['label_15_real'] = 0
         sample['label_15_weight'] = 0
         sample['label_15_raw'] = 0
         sample['label_15_raw_real'] = 0
         sample['label_15_raw_weight'] = 0
         sample.to_csv('E:/pythonProject/future/data/datafile/prediction_sample/{model_name}/prediction_sample_{date}.csv'.format(model_name=self.model_name, date=str(self.date_end)), mode='a',header=True, index=False, encoding='utf-8')
      return feature_all
if __name__ == '__main__':
   years = [2023]
   is_predict = False
   model_name = 'model_v9'
   date_start = '2023-02-16'
   date_end = '2023-02-16'
   # years = [2008, 2009]
   years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,2021,2022]

   # time.sleep(18000)
   for year in years:
      path = 'E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_'
      quater_path = 'E:/pythonProject/future/data/datafile/code_quarter_data_v2_all.csv'
      output_path = 'E:/pythonProject/future/data/datafile/feature/{year}_feature_v9.csv'.format(year=str(year))
      # raw_k_data = pd.read_csv(path + str(year) + '.csv')
      # raw_k_data.to_csv('E:/pythonProject/future/data/datafile/raw_feature/test_code_k_data_v5_' + str(year) + '.csv', mode='a', header=True, index=False)
      feature = Feature(path, year, quater_path, is_predict, date_start, date_end, model_name)
      feature_all = feature.feature_process()
      if os.path.isfile(output_path):
         feature_all.to_csv(output_path, mode='a', header=False, index=False)
      else:
         feature_all.to_csv(output_path, mode='w', header=True, index=False)
      del feature_all
      gc.collect()