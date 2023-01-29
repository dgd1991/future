import numpy as np
import pandas as pd
from tools.path_enum import Path


pd.set_option('display.max_columns', 200)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)

class ModelEval(object):
	def __init__(self, date, model_name):
		self.date = date
		self.model_name = model_name
		self.sw_code_all_industry_path = Path.sw_code_all_industry_path

	def eval(self):
		df = pd.read_csv('E:/pythonProject/future/data/datafile/prediction_result/{model_name}/prediction_result_{date}.csv'.format(model_name=self.model_name, date=str(self.date)), encoding='utf-8')

		df.columns = ['date', 'code', 'industry_name_level1', 'industry_name_level2', 'industry_name_level3', 'predition', 'label_7', 'label_7_real', 'label_7_weight', 'label_7_max', 'label_7_max_real', 'label_7_max_weight', 'label_15', 'label_15_real', 'label_15_weight', 'label_15_max', 'label_15_max_real', 'label_15_max_weight']
		df['code'] = df['code'].map(lambda x: x.replace('b', '').replace("'", ''))
		df = df[['date', 'code', 'predition']]
		self.code_industry = pd.read_csv(self.sw_code_all_industry_path, encoding='utf-8')
		self.code_industry['start_date'] = pd.to_datetime(self.code_industry['start_date'])
		self.code_industry['row_num'] = self.code_industry.groupby(['code'])['start_date'].rank(ascending=False, method='first').astype(int)
		self.code_industry = self.code_industry[self.code_industry['row_num'] == 1]
		self.code_industry = self.code_industry[['code', 'industry_name_level1', 'industry_name_level2', 'industry_name_level3']]

		df = pd.merge(df, self.code_industry,  how="left", left_on=['code'], right_on=['code'])
		df['industry_name_level1'] = df['industry_name_level1'].map(lambda x: x if type(x) == str else 'level1_default')
		df['industry_name_level2'] = df['industry_name_level2'].map(lambda x: x if type(x) == str else 'level2_default')
		df['industry_name_level3'] = df['industry_name_level3'].map(lambda x: x if type(x) == str else 'level3_default')

		# # label_7
		df_head30 = df.sort_values(['date', 'predition'], ascending=[True, False]).groupby('date').head(30)
		df_head30.to_csv('E:/pythonProject/future/data/datafile/prediction_result/{model_name}/result_top30_{date}.csv'.format(model_name=self.model_name, date=str(self.date)), mode='a',header=True, index=False, encoding='utf-8')
		industry1 = df[['date', 'industry_name_level1', 'predition']].groupby(['date', 'industry_name_level1']).mean()
		industry1 = industry1.reset_index(level=0, drop=False)
		industry1 = industry1.reset_index(level=0, drop=False)
		industry1 = industry1.sort_values(['date', 'predition'], ascending=[True, False]).head(5)
		industry2 = df[['date', 'industry_name_level2', 'predition']].groupby(['date', 'industry_name_level2']).mean()
		industry2 = industry2.reset_index(level=0, drop=False)
		industry2 = industry2.reset_index(level=0, drop=False)
		industry2 = industry2.sort_values(['date', 'predition'], ascending=[True, False]).head(5)
		industry3 = df[['date', 'industry_name_level3', 'predition']].groupby(['date', 'industry_name_level3']).mean()
		industry3 = industry3.reset_index(level=0, drop=False)
		industry3 = industry3.reset_index(level=0, drop=False)
		industry3 = industry3.sort_values(['date', 'predition'], ascending=[True, False]).head(5)

		industry1.to_csv('E:/pythonProject/future/data/datafile/prediction_result/{model_name}/indus_top5_{date}.csv'.format(model_name=self.model_name, date=str(self.date)), mode='a',header=True, index=False, encoding='utf-8')
		industry2.to_csv('E:/pythonProject/future/data/datafile/prediction_result/{model_name}/indus_top5_{date}.csv'.format(model_name=self.model_name, date=str(self.date)), mode='a',header=True, index=False, encoding='utf-8')
		industry3.to_csv('E:/pythonProject/future/data/datafile/prediction_result/{model_name}/indus_top5_{date}.csv'.format(model_name=self.model_name, date=str(self.date)), mode='a',header=True, index=False, encoding='utf-8')

	# print('all_mean: ' + str(all_mean))
if __name__ == "__main__":
	# , 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019
	dates = ["2023-01-20"]
	# years = [2009, 2010]
	model_name = 'saved_model_v7'
	result = []
	for date in dates:
		modelEval = ModelEval(date, model_name)
		modelEval.eval()
	# print(result_all)
	# result_all.to_csv('E:/pythonProject/future/data/datafile/prediction_result/{model_name}/result_{year}.csv'.format(model_name=model_name,year=str('256b_3e')), mode='a',header=True, index=False, encoding='utf-8')