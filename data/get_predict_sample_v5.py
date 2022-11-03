import copy
import gc
import time

import numpy as np
import pandas as pd
from feature.feature_process import get_classification_label
# 省略问题修改配置, 打印100列数据
pd.set_option('display.max_columns', 200)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)

class Sample(object):
	def __init__(self, year, date, base_path, model_name):
		self.year = year
		self.fea_path = base_path + '/feature/' + str(year) + '_feature_v5.csv'
		self.label_path = base_path + '/label/' + str(year) + '_raw_v4.csv'
		self.year = year
		self.date = date
		self.output_dir = base_path
		self.model_name = model_name
	def get_sample(self):
		feature = pd.read_csv(self.fea_path)
		feature['date'] = feature['date'].map(lambda x: int(x.replace('-', '')))
		# if self.year == 2008:
		# 	feature['date'] = pd.to_datetime(feature["date"], errors='coerce')
		sample = feature
		sample = sample[sample['date'] == self.date]
		sample = sample[sample['code_market'] != 4]
		sample['label_7'] = 0
		sample['label_7_real'] = 0
		sample['label_7_weight'] = 0
		sample['label_7_max'] = 0
		sample['label_7_max_real'] = 0
		sample['label_7_max_weight'] = 0
		sample['label_15'] = 0
		sample['label_15_real'] = 0
		sample['label_15_weight'] = 0
		sample['label_15_max'] = 0
		sample['label_15_max_real'] = 0
		sample['label_15_max_weight'] = 0

		del feature
		gc.collect()
		sample = sample.sort_values(by=['date', 'code'], ascending=True)
		sample.to_csv('{output_dir}/prediction_sample/{model_name}/prediction_sample_{date}.csv'.format(output_dir=self.output_dir, model_name=self.model_name, date=str(self.date)), mode='a',header=True, index=False, encoding='utf-8')
if __name__ == '__main__':
	base_path = 'E:/pythonProject/future/data/datafile'
	model_name = 'model_v5'
	year = 2022
	date = 20221102
	sample = Sample(year, date, base_path, model_name)
	sample.get_sample()
