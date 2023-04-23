import copy
import gc
import time

import numpy as np
import pandas as pd
from feature.feature_process import get_classification_label
from tools.Tools import Tools

# 省略问题修改配置, 打印100列数据
pd.set_option('display.max_columns', 200)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)

class Sample(object):
	def __init__(self, year, base_path, model_name):
		self.year = year
		self.code_fea_path = base_path + '/feature/' + model_name + '/code_feature_' + str(year) + '.csv'
		self.label_path = base_path + '/label/' + str(year) + '_raw_v6.csv'
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
		"code,date,pctChg_7,sh_pctChg_7,lable_7_rank,lable_7_count,pctChg_15,sh_pctChg_15,lable_15_rank,lable_15_count"

		# label = label[['code','date','pctChg_7','zs_pctChg_7','pctChg_15','zs_pctChg_15']]
		label.dropna(axis=0, inplace=True)
		label['label_7'] = label['lable_7_rank'] / label['lable_7_count']
		label['label_7'] = label['label_7'].map(lambda x: 1 if x<=0.1 else 0)

		label['label_15'] = label['lable_15_rank'] / label['lable_15_count']
		label['label_15'] = label['label_15'].map(lambda x: 1 if x<=0.1 else 0)

		label = label[['code','date','label_7','pctChg_7','sh_pctChg_7','label_15','pctChg_15','sh_pctChg_15']]

		feature = pd.read_csv(self.code_fea_path)
		feature['date'] = feature['date'].map(lambda x: int(x.replace('-', '')))
		# if self.year == 2008:
		# 	feature['date'] = pd.to_datetime(feature["date"], errors='coerce')
		feature = pd.merge(feature, label, how="inner", left_on=['date', "code"],right_on=['date', 'code'])
		del label
		gc.collect()
		feature = feature.round(6)
		feature.to_csv('{output_dir}/sample/{model_name}/train_sample_{year}.csv'.format(output_dir=self.output_dir, model_name=self.model_name, year=str(self.year)), mode='w',header=True, index=False, encoding='utf-8')
		del feature
		gc.collect()
if __name__ == '__main__':
	base_path = 'E:/pythonProject/future/data/datafile'
	model_name = 'model_v12'
	# time.sleep(10000)
	years = [2023]
	# years = [2008,2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
	for year in years:
		sample = Sample(year, base_path, model_name)
		sample.get_sample()
		# time.sleep(600)
