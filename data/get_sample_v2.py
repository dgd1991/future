import copy
import gc
import pandas as pd
from feature.feature_process import get_classification_label

class Sample(object):
	def __init__(self, year, base_path):
		self.year = year
		self.fea_path = base_path + '/feature/' + str(year) + '_feature.csv'
		self.label_path = base_path + '/label/' + str(year) + '_raw.csv'
		self.year = year
		self.output_dir = base_path
	def get_sample(self):
		label = pd.read_csv(self.label_path)
		label.dropna(axis=0,how='any')
		feature = pd.read_csv(self.fea_path)
		sample = pd.merge(feature, label, how="right", left_on=['code', "date"],right_on=['code', 'date'])
		del feature
		del label
		gc.collect()
		sample = sample.sort_values(by=['date', 'code'], ascending=True)
		sample.to_csv('{output_dir}/sample/train_sample_{year}.csv'.format(output_dir=self.output_dir, year=str(self.year)), mode='a',header=True, index=False, encoding='utf-8')

if __name__ == '__main__':
	base_path = 'E:/pythonProject/future/data/datafile'
	years = [2020,2021]
	for year in years:
		sample = Sample(year, base_path)
		sample.get_sample()
