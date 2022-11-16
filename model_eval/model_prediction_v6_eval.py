import numpy as np
import pandas as pd


pd.set_option('display.max_columns', 200)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)

class ModelEval(object):
	def __init__(self, date, model_name):
		self.date = date
		self.model_name = model_name

	def eval(self):
		df = pd.read_csv('E:/pythonProject/future/data/datafile/prediction_result/{model_name}/prediction_result_{date}.csv'.format(model_name=self.model_name, date=str(self.date)), encoding='utf-8')

		df.columns = ['date', 'code', 'industry_name_level1', 'industry_name_level2', 'industry_name_level3', 'predition', 'label_7', 'label_7_real', 'label_7_weight', 'label_7_max', 'label_7_max_real', 'label_7_max_weight', 'label_15', 'label_15_real', 'label_15_weight', 'label_15_max', 'label_15_max_real', 'label_15_max_weight']
		df['code'] = df['code'].map(lambda x: x.replace('b', '').replace("'", ''))

		# df = df[df['code'].map(lambda x: True if x.startwith('sh.68') else False)]

		# # label_7
		df_head20 = df.sort_values(['date', 'predition'], ascending=[True, False]).groupby('date').head(30)
		df_head20.to_csv('E:/pythonProject/future/data/datafile/prediction_result/{model_name}/prediction_result_top20_{date}.csv'.format(model_name=self.model_name, date=str(self.date)), mode='a',header=True, index=False, encoding='utf-8')
		df_head20['result'] = (2 * df_head20['label_7'] - 1) * df_head20['label_7_weight']
		df['result'] = (2 * df['label_7'] - 1) * df['label_7_weight']
		all_mean = df[['label_7', 'result', 'label_7_real']].mean()
		all_code_mean = df_head20[['label_7', 'result', 'label_7_real']].mean()

		# label_15
		# df_head20 = df.sort_values(['date', 'predition'], ascending=[True, False]).groupby('date').head(20)
		# df_head20['result'] = (2 * df_head20['label_15'] - 1) * df_head20['label_15_weight']
		# # all_mean = df_head20[['date', 'label_15', 'result']].groupby('date').mean()
		# df['result'] = (2 * df['label_15'] - 1) * df['label_15_weight']
		# all_mean = df[['label_15', 'result', 'label_15_real']].mean()
		# all_code_mean = df_head20[['label_15', 'result', 'label_15_real']].mean()


		all_code_mean['date'] = self.date
		all_code_mean = pd.DataFrame([dict(zip(all_code_mean.index, all_code_mean.values.T))])
		all_mean['date'] = self.date
		all_mean = pd.DataFrame([dict(zip(all_mean.index, all_mean.values.T))])
		return all_mean, all_code_mean

	# print('all_mean: ' + str(all_mean))
if __name__ == "__main__":
	# , 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019
	dates = [20221116]
	# years = [2009, 2010]
	model_name = 'saved_model_v6'
	result = []
	for date in dates:
		modelEval = ModelEval(date, model_name)
		all_mean, all_code_mean = modelEval.eval()
		result.append(all_code_mean)
		result.append(all_mean)

	result_all = pd.concat(result, axis=0)
	# print(result_all)
	# result_all.to_csv('E:/pythonProject/future/data/datafile/prediction_result/{model_name}/result_{year}.csv'.format(model_name=model_name,year=str('256b_3e')), mode='a',header=True, index=False, encoding='utf-8')