import copy
import gc
import time

import numpy as np
import pandas as pd
from feature.feature_process import get_classification_label
from feature.feature_process import *
from tools.Tools import Tools
# 省略问题修改配置, 打印100列数据
pd.set_option('display.max_columns', 200)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)

class Sample(object):
	def __init__(self, year, base_path, model_name):
		self.year = year
		self.fea_path = base_path + '/feature/' + str(year) + '_feature_v9.csv'
		self.label_path = base_path + '/label/' + str(year) + '_raw_v5.csv'
		self.year = year
		self.output_dir = base_path
		self.model_name = model_name
		self.tools = Tools()
	def get_sample(self):
		label = pd.read_csv(self.label_path)
		if self.year == 2008:
			label['date'] = pd.to_numeric(label["date"], errors='coerce')
			label = label[label['date']>20080702]
		# label.dropna(axis=0, inplace=True)
		label['zs_pctChg_7'] = label.apply(lambda x: x.cy_pctChg_7 if self.tools.code_market(x.code)==3 else (x.sh_pctChg_7 if self.tools.code_market(x.code)==1 else (x.sz_pctChg_7 if self.tools.code_market(x.code)==2 else np.nan)), axis=1)
		label['zs_pctChg_15'] = label.apply(lambda x: x.cy_pctChg_15 if self.tools.code_market(x.code)==3 else (x.sh_pctChg_15 if self.tools.code_market(x.code)==1 else (x.sz_pctChg_15 if self.tools.code_market(x.code)==2 else np.nan)), axis=1)
		label = label[['code','date','pctChg_7','zs_pctChg_7','pctChg_15','zs_pctChg_15']]
		label.dropna(axis=0, inplace=True)
		label['label_7'] = label['pctChg_7'] - label['zs_pctChg_7'] - 0.075
		label['label_7_real'] = label['pctChg_7'] - label['zs_pctChg_7']
		label['label_7_weight'] = label['label_7'].map(lambda x: min(abs(x*100), 2.5)*1.0/2.5)
		label['label_7'] = label['label_7'].map(lambda x: 1 if x>0 else 0)

		label['label_7_raw'] = label['pctChg_7'] - 0.075
		label['label_7_raw_real'] = label['pctChg_7']
		label['label_7_raw_weight'] = label['label_7_raw'].map(lambda x: min(abs(x * 100), 2.5) * 1.0 / 2.5)
		label['label_7_raw'] = label['label_7_raw'].map(lambda x: 1 if x > 0 else 0)

		label['label_15'] = label['pctChg_15'] - label['zs_pctChg_15'] - 0.1
		label['label_15_real'] = label['pctChg_15'] - label['zs_pctChg_15']
		label['label_15_weight'] = label['label_15'].map(lambda x: min(abs(x*100), 5)*1.0/5)
		label['label_15'] = label['label_15'].map(lambda x: 1 if x>0 else 0)

		label['label_15_raw'] = label['pctChg_15'] - 0.1
		label['label_15_raw_real'] = label['pctChg_15']
		label['label_15_raw_weight'] = label['label_15_raw'].map(lambda x: min(abs(x * 100), 5) * 1.0 / 5)
		label['label_15_raw'] = label['label_15_raw'].map(lambda x: 1 if x > 0 else 0)


		label = label[['code','date','label_7','label_7_real','label_7_weight','label_7_raw','label_7_raw_real','label_7_raw_weight','label_15','label_15_real','label_15_weight','label_15_raw','label_15_raw_real','label_15_raw_weight']]

		feature = pd.read_csv(self.fea_path)
		feature['date'] = feature['date'].map(lambda x: int(x.replace('-', '')))
		# if self.year == 2008:
		# 	feature['date'] = pd.to_datetime(feature["date"], errors='coerce')
		sample = pd.merge(feature, label, how="inner", left_on=['code', "date"],right_on=['code', 'date'])
		if self.year == 2010:
			sample = sample[sample['code_market'] != 3]
		del feature
		del label
		gc.collect()
		sample = sample.sort_values(by=['date', 'code'], ascending=True).round(5)
		sample.to_csv('{output_dir}/sample/{model_name}/train_sample_{year}.csv'.format(output_dir=self.output_dir, model_name=self.model_name, year=str(self.year)), mode='w',header=True, index=False, encoding='utf-8')
if __name__ == '__main__':
	base_path = 'E:/pythonProject/future/data/datafile'
	model_name = 'model_v9'
	years = [2021, 2022]
	# years = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
	for year in years:
		sample = Sample(year, base_path, model_name)
		sample.get_sample()
		# time.sleep(600)
