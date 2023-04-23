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
	def __init__(self, year, date, base_path, model_name):
		self.code_fea_path = base_path + '/feature/' + model_name + '/code_feature_' + str(year) + '.csv'
		self.label_path = base_path + '/label/' + str(year) + '_raw_v6.csv'
		self.date = date
		self.output_dir = base_path
		self.model_name = model_name
		self.tools = Tools()
	def get_sample(self):
		feature = pd.read_csv(self.code_fea_path)
		feature = feature[feature['date'] == self.date]
		feature['date'] = feature['date'].map(lambda x: int(x.replace('-', '')))
		feature['label_7'] = 0
		feature['pctChg_7'] = 0
		feature['sh_pctChg_7'] = 0
		feature['label_15'] = 0
		feature['pctChg_15'] = 0
		feature['sh_pctChg_15'] = 0

		feature = feature.round(6)
		feature.to_csv('{output_dir}/prediction_sample/{model_name}/prediction_sample_{date}.csv'.format(output_dir=self.output_dir, model_name=self.model_name, date=str(self.date)), mode='w',header=True, index=False, encoding='utf-8')
		del feature
		gc.collect()
if __name__ == '__main__':
	base_path = 'E:/pythonProject/future/data/datafile'
	model_name = 'model_v12'
	year = 2023
	date = '2023-04-17'
	sample = Sample(year, date, base_path, model_name)
	sample.get_sample()
