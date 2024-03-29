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

from tools.path_enum import Path

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
      self.sw_code_all_industry_path = Path.sw_code_all_industry_path

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
      raw_k_data = raw_k_data[(raw_k_data['industry_id_level3'] > 0)]
      raw_k_data_his = pd.read_csv(self.k_file_path_his)
      raw_k_data_his = raw_k_data_his[(raw_k_data_his['industry_id_level3'] > 0)]
      if is_predict:
         raw_k_data_his = raw_k_data_his[raw_k_data_his['date']>self.tools.get_recent_month_date(self.date_start, -14)]
      raw_k_data = pd.concat([raw_k_data_his, raw_k_data], axis=0)
      del raw_k_data_his
      gc.collect()
      raw_k_data['date'] = raw_k_data['date'].map(lambda x: int(x.replace('-', '')))
      raw_k_data["tradestatus"] = pd.to_numeric(raw_k_data["tradestatus"], errors='coerce')
      raw_k_data["turn"] = pd.to_numeric(raw_k_data["turn"], errors='coerce')
      raw_k_data["close"] = pd.to_numeric(raw_k_data["close"], errors='coerce')
      raw_k_data["pctChg"] = pd.to_numeric(raw_k_data["pctChg"], errors='coerce')
      raw_k_data['amount'] = 0.00000001 * pd.to_numeric(raw_k_data["amount"], errors='coerce')
      raw_k_data = raw_k_data[(raw_k_data['tradestatus'] == 1) & (raw_k_data['turn'] > 0) & (raw_k_data['pctChg'] < 21) & (raw_k_data['pctChg'] > -21)]


      raw_k_data = raw_k_data[["code","date","turn", "close", "amount"]]
      raw_k_data['date_new'] = raw_k_data['date']
      raw_k_data = raw_k_data.sort_values(['code', 'date'])
      raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))
      tmp = raw_k_data[['date_new','close']]

      raw_k_data['max_turn'] = raw_k_data['turn'].groupby(level=0).apply(lambda x: x.rolling(min_periods=10, window=10, center=False).apply(lambda y: max(y[0:9])))
      raw_k_data['turn_diff'] = raw_k_data['turn']/raw_k_data['max_turn']
      raw_k_data['min_close'] = raw_k_data['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=30, window=30, center=False).apply(lambda y: min(y[0:29])))
      raw_k_data['close_diff'] = raw_k_data['close'] / raw_k_data['min_close']

      tmp['close_cur_day'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=8, window=8, center=False).apply(lambda y: y[0]))
      tmp['date_7'] = tmp['date_new'].groupby(level=0).apply(lambda x: x.rolling(min_periods=8, window=8, center=False).apply(lambda y: y[0]))
      tmp['pctChg_7'] = (tmp['close'] - tmp['close_cur_day'])/tmp['close_cur_day']

      raw_k_data = raw_k_data.reset_index(level=0, drop=True)
      raw_k_data = raw_k_data.reset_index(level=0, drop=False)

      tmp = tmp.reset_index(level=0, drop=False)
      tmp = tmp.reset_index(level=0, drop=False)
      tmp = tmp[["code","date_7","pctChg_7"]]

      raw_k_data = pd.merge(raw_k_data, tmp, how="inner", left_on=["code","date"], right_on=["code","date_7"])

      raw_k_data = raw_k_data.sort_values(['turn_diff'],ascending=False)

      self.code_industry = pd.read_csv(self.sw_code_all_industry_path, encoding='utf-8')
      self.code_industry['start_date'] = pd.to_datetime(self.code_industry['start_date'])
      self.code_industry['row_num'] = self.code_industry.groupby(['code'])['start_date'].rank(ascending=False, method='first').astype(int)
      self.code_industry = self.code_industry[self.code_industry['row_num'] == 1]
      self.code_industry = self.code_industry[['code', 'industry_name_level1', 'industry_name_level2', 'industry_name_level3']]
      raw_k_data = pd.merge(raw_k_data, self.code_industry, how="inner", left_on=['code'], right_on=['code'])

      tmp1 = raw_k_data.groupby(["date","industry_name_level1"])['code'].count().sort_values(ascending=False).reset_index()
      tmp1.columns = ["date",'industry_name_level1', 'level1_cnt']
      tmp2 = raw_k_data.groupby(["date","industry_name_level2"])['code'].count().sort_values(ascending=False).reset_index()
      tmp2.columns = ["date",'industry_name_level2', 'level2_cnt']
      tmp3 = raw_k_data.groupby(["date","industry_name_level3"])['code'].count().sort_values(ascending=False).reset_index()
      tmp3.columns = ["date",'industry_name_level3', 'level3_cnt']
      raw_k_data = pd.merge(raw_k_data, tmp1, how="inner", left_on=["date",'industry_name_level1'], right_on=["date",'industry_name_level1'])
      raw_k_data = pd.merge(raw_k_data, tmp2, how="inner", left_on=["date",'industry_name_level2'], right_on=["date",'industry_name_level2'])
      raw_k_data = pd.merge(raw_k_data, tmp3, how="inner", left_on=["date",'industry_name_level3'], right_on=["date",'industry_name_level3'])
      raw_k_data = raw_k_data[(raw_k_data['level1_cnt']>10) | (raw_k_data['level2_cnt']>5) | (raw_k_data['level3_cnt']>3)]
      raw_k_data = raw_k_data.sort_values(by=['date'], ascending=False)


      if is_predict:
         raw_k_data = raw_k_data[(raw_k_data['date'] >= self.date_start) & (raw_k_data['date'] <= self.date_end)]
         raw_k_data.to_csv('E:/pythonProject/future/data/datafile/prediction_sample/{model_name}/test_{date}.csv'.format(model_name=self.model_name, date=str(self.date_end)), mode='w',header=True, index=False, encoding='utf-8')
      else:
         raw_k_data.to_csv('E:/pythonProject/future/data/datafile/prediction_sample/{model_name}/test_{date}.csv'.format(model_name=self.model_name, date=str(self.year)), mode='w',header=True, index=False, encoding='utf-8')

      # raw_k_data = raw_k_data[raw_k_data['turn_diff']>2]
      # print(raw_k_data.groupby("industry_name_level1")['code'].count().sort_values(ascending=False).reset_index())
      # print(raw_k_data.groupby("industry_name_level2")['code'].count().sort_values(ascending=False).reset_index())
      # print(raw_k_data.groupby("industry_name_level3")['code'].count().sort_values(ascending=False).reset_index())
      return raw_k_data
if __name__ == '__main__':
   years = [2023]
   is_predict = False
   model_name = 'model_v12'
   date_start = '2023-04-19'
   date_end = '2023-04-19'
   # years = [2008, 2009]
   # years = [2020,2021,2022,2023]
   # years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,2021,2022]

   # time.sleep(18000)
   for year in years:
      path = 'E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_'
      quater_path = 'E:/pythonProject/future/data/datafile/code_quarter_data_v2_all.csv'
      output_path = 'E:/pythonProject/future/data/datafile/feature/{year}_feature_v9.csv'.format(year=str(year))
      # raw_k_data = pd.read_csv(path + str(year) + '.csv')
      # raw_k_data.to_csv('E:/pythonProject/future/data/datafile/raw_feature/test_code_k_data_v5_' + str(year) + '.csv', mode='a', header=True, index=False)
      feature = Feature(path, year, quater_path, is_predict, date_start, date_end, model_name)
      feature_all = feature.feature_process()
      # if os.path.isfile(output_path):
      #    feature_all.to_csv(output_path, mode='a', header=False, index=False)
      # else:
      #    feature_all.to_csv(output_path, mode='w', header=True, index=False)
      # del feature_all
      # gc.collect()