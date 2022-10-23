import copy
import gc
import pandas as pd
from feature.feature_process import get_classification_label

class Sample(object):
	def __init__(self, year, base_path, model_name):
		self.year = year
		self.fea_path = base_path + '/feature/' + str(year) + '_feature.csv'
		self.label_path = base_path + '/label/' + str(year) + '_raw.csv'
		self.year = year
		self.output_dir = base_path
		self.model_name = model_name
	def get_sample(self):
		label = pd.read_csv(self.label_path)
		if self.year == 2008:
			label['date'] = pd.to_datetime(label["date"], errors='coerce')
			label = label[label['date']>'2008-07-02']
		label.dropna(axis=0,how='any')
		label['label_7d'] = pd.to_numeric(label["label_7d"], errors='coerce')
		label['label'] = label['label_7d'].map(lambda x: 1 if x>0 else 0)
		label['label_weight'] = label['label_7d'].map(lambda x: min(abs(x), 10)*1.0/10)
		feature = pd.read_csv(self.fea_path)
		if self.year == 2008:
			feature['date'] = pd.to_datetime(feature["date"], errors='coerce')
		sample = pd.merge(feature, label, how="right", left_on=['code', "date"],right_on=['code', 'date'])
		del feature
		del label
		gc.collect()
		sample = sample.sort_values(by=['date', 'code'], ascending=True)
		sample.to_csv('{output_dir}/sample/{model_name}/train_sample_{year}.csv'.format(output_dir=self.output_dir, model_name=self.model_name, year=str(self.year)), mode='a',header=True, index=False, encoding='utf-8')

if __name__ == '__main__':
	base_path = 'E:/pythonProject/future/data/datafile'
	model_name = 'model_v3'
	years = [2019, 2020, 2021]
	for year in years:
		sample = Sample(year, base_path, model_name)
		sample.get_sample()
