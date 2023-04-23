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
		self.industry1_fea_path = base_path + '/feature/' + model_name + '/industry1_feature_' + str(year) + '.csv'
		self.industry2_fea_path = base_path + '/feature/' + model_name + '/industry2_feature_' + str(year) + '.csv'
		self.industry3_fea_path = base_path + '/feature/' + model_name + '/industry3_feature_' + str(year) + '.csv'
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

		industry1_feature = pd.read_csv(self.industry1_fea_path)
		industry1_col_name = ['date', 'industry_id_level1', 'industry_id_level1_open_ratio', 'industry_id_level1_close_ratio', 'industry_id_level1_high_ratio', 'industry_id_level1_low_ratio', 'industry_id_level1_pctChg', 'industry_id_level1_rise_ratio', 'industry_id_level1_close', 'industry_id_level1_preclose', 'industry_id_level1_open', 'industry_id_level1_high', 'industry_id_level1_low', 'industry_id_level1_open_ratio_7d_avg', 'industry_id_level1_close_ratio_7d_avg', 'industry_id_level1_high_ratio_7d_avg', 'industry_id_level1_low_ratio_7d_avg', 'industry_id_level1_rsv_5', 'industry_id_level1_k_value_5', 'industry_id_level1_d_value_5', 'industry_id_level1_j_value_5', 'industry_id_level1_rsv_9', 'industry_id_level1_k_value_9', 'industry_id_level1_d_value_9', 'industry_id_level1_j_value_9', 'industry_id_level1_rsv_19', 'industry_id_level1_k_value_19', 'industry_id_level1_d_value_19', 'industry_id_level1_j_value_19', 'industry_id_level1_rsv_73', 'industry_id_level1_k_value_73', 'industry_id_level1_d_value_73', 'industry_id_level1_j_value_73', 'industry_id_level1_macd_dif_ratio', 'industry_id_level1_macd_positive_ratio_2', 'industry_id_level1_macd_positive_ratio_3', 'industry_id_level1_macd_positive_ratio_5', 'industry_id_level1_macd_positive_ratio_10', 'industry_id_level1_macd_positive_ratio_20', 'industry_id_level1_macd_positive_ratio_40', 'industry_id_level1_width_2', 'industry_id_level1_close_mb2_diff', 'industry_id_level1_width_3', 'industry_id_level1_close_mb3_diff', 'industry_id_level1_width_5', 'industry_id_level1_close_mb5_diff', 'industry_id_level1_width_10', 'industry_id_level1_close_mb10_diff', 'industry_id_level1_width_20', 'industry_id_level1_close_mb20_diff', 'industry_id_level1_width_40', 'industry_id_level1_close_mb40_diff', 'industry_id_level1_cr_bias_26d', 'industry_id_level1_rsi_3d', 'industry_id_level1_rsi_5d', 'industry_id_level1_rsi_10d', 'industry_id_level1_rsi_20d', 'industry_id_level1_rsi_40d', 'industry_id_level1_turn_3d_avg', 'industry_id_level1_turn_rank_3d_avg', 'industry_id_level1_turn_3dmax_dif', 'industry_id_level1_turn_3dmin_dif', 'industry_id_level1_turn_5d_avg', 'industry_id_level1_turn_rank_5d_avg', 'industry_id_level1_turn_5dmax_dif', 'industry_id_level1_turn_5dmin_dif', 'industry_id_level1_turn_10d_avg', 'industry_id_level1_turn_rank_10d_avg', 'industry_id_level1_turn_10dmax_dif', 'industry_id_level1_turn_10dmin_dif', 'industry_id_level1_turn_20d_avg', 'industry_id_level1_turn_rank_20d_avg', 'industry_id_level1_turn_20dmax_dif', 'industry_id_level1_turn_20dmin_dif', 'industry_id_level1_turn_30d_avg', 'industry_id_level1_turn_rank_30d_avg', 'industry_id_level1_turn_30dmax_dif', 'industry_id_level1_turn_30dmin_dif', 'industry_id_level1_turn_60d_avg', 'industry_id_level1_turn_rank_60d_avg', 'industry_id_level1_turn_60dmax_dif', 'industry_id_level1_turn_60dmin_dif', 'industry_id_level1_turn_120d_avg', 'industry_id_level1_turn_rank_120d_avg', 'industry_id_level1_turn_120dmax_dif', 'industry_id_level1_turn_120dmin_dif', 'industry_id_level1_turn_240d_avg', 'industry_id_level1_turn_rank_240d_avg', 'industry_id_level1_turn_240dmax_dif', 'industry_id_level1_turn_240dmin_dif', 'industry_id_level1_pctChg_3d', 'industry_id_level1_pctChg_5d', 'industry_id_level1_pctChg_10d', 'industry_id_level1_pctChg_20d', 'industry_id_level1_pctChg_30d', 'industry_id_level1_pctChg_60d', 'industry_id_level1_pctChg_120d', 'industry_id_level1_pctChg_240d', 'industry_id_level1_pctChg_rank_ratio3d', 'industry_id_level1_pctChg_rank_ratio5d', 'industry_id_level1_pctChg_rank_ratio10d', 'industry_id_level1_pctChg_rank_ratio20d', 'industry_id_level1_pctChg_rank_ratio30d', 'industry_id_level1_pctChg_rank_ratio60d', 'industry_id_level1_pctChg_rank_ratio120d', 'industry_id_level1_pctChg_rank_ratio240d']
		industry1_feature = industry1_feature[industry1_col_name]
		industry1_feature['date'] = industry1_feature['date'].map(lambda x: int(x.replace('-', '')))

		feature = pd.read_csv(self.code_fea_path)
		feature['date'] = feature['date'].map(lambda x: int(x.replace('-', '')))
		feature = pd.merge(feature, industry1_feature, how="left", left_on=['date', "industry_id_level1"],right_on=['date', 'industry_id_level1'])
		del industry1_feature
		gc.collect()

		industry2_feature = pd.read_csv(self.industry2_fea_path)
		industry2_col_name = ['date', 'industry_id_level2', 'industry_id_level2_open_ratio', 'industry_id_level2_close_ratio', 'industry_id_level2_high_ratio', 'industry_id_level2_low_ratio', 'industry_id_level2_pctChg', 'industry_id_level2_rise_ratio', 'industry_id_level2_close', 'industry_id_level2_preclose', 'industry_id_level2_open', 'industry_id_level2_high', 'industry_id_level2_low', 'industry_id_level2_open_ratio_7d_avg', 'industry_id_level2_close_ratio_7d_avg', 'industry_id_level2_high_ratio_7d_avg', 'industry_id_level2_low_ratio_7d_avg', 'industry_id_level2_rsv_5', 'industry_id_level2_k_value_5', 'industry_id_level2_d_value_5', 'industry_id_level2_j_value_5', 'industry_id_level2_rsv_9', 'industry_id_level2_k_value_9', 'industry_id_level2_d_value_9', 'industry_id_level2_j_value_9', 'industry_id_level2_rsv_19', 'industry_id_level2_k_value_19', 'industry_id_level2_d_value_19', 'industry_id_level2_j_value_19', 'industry_id_level2_rsv_73', 'industry_id_level2_k_value_73', 'industry_id_level2_d_value_73', 'industry_id_level2_j_value_73', 'industry_id_level2_macd_dif_ratio', 'industry_id_level2_macd_positive_ratio_2', 'industry_id_level2_macd_positive_ratio_3', 'industry_id_level2_macd_positive_ratio_5', 'industry_id_level2_macd_positive_ratio_10', 'industry_id_level2_macd_positive_ratio_20', 'industry_id_level2_macd_positive_ratio_40', 'industry_id_level2_width_2', 'industry_id_level2_close_mb2_diff', 'industry_id_level2_width_3', 'industry_id_level2_close_mb3_diff', 'industry_id_level2_width_5', 'industry_id_level2_close_mb5_diff', 'industry_id_level2_width_10', 'industry_id_level2_close_mb10_diff', 'industry_id_level2_width_20', 'industry_id_level2_close_mb20_diff', 'industry_id_level2_width_40', 'industry_id_level2_close_mb40_diff', 'industry_id_level2_cr_bias_26d', 'industry_id_level2_rsi_3d', 'industry_id_level2_rsi_5d', 'industry_id_level2_rsi_10d', 'industry_id_level2_rsi_20d', 'industry_id_level2_rsi_40d', 'industry_id_level2_turn_3d_avg', 'industry_id_level2_turn_rank_3d_avg', 'industry_id_level2_turn_3dmax_dif', 'industry_id_level2_turn_3dmin_dif', 'industry_id_level2_turn_5d_avg', 'industry_id_level2_turn_rank_5d_avg', 'industry_id_level2_turn_5dmax_dif', 'industry_id_level2_turn_5dmin_dif', 'industry_id_level2_turn_10d_avg', 'industry_id_level2_turn_rank_10d_avg', 'industry_id_level2_turn_10dmax_dif', 'industry_id_level2_turn_10dmin_dif', 'industry_id_level2_turn_20d_avg', 'industry_id_level2_turn_rank_20d_avg', 'industry_id_level2_turn_20dmax_dif', 'industry_id_level2_turn_20dmin_dif', 'industry_id_level2_turn_30d_avg', 'industry_id_level2_turn_rank_30d_avg', 'industry_id_level2_turn_30dmax_dif', 'industry_id_level2_turn_30dmin_dif', 'industry_id_level2_turn_60d_avg', 'industry_id_level2_turn_rank_60d_avg', 'industry_id_level2_turn_60dmax_dif', 'industry_id_level2_turn_60dmin_dif', 'industry_id_level2_turn_120d_avg', 'industry_id_level2_turn_rank_120d_avg', 'industry_id_level2_turn_120dmax_dif', 'industry_id_level2_turn_120dmin_dif', 'industry_id_level2_turn_240d_avg', 'industry_id_level2_turn_rank_240d_avg', 'industry_id_level2_turn_240dmax_dif', 'industry_id_level2_turn_240dmin_dif', 'industry_id_level2_pctChg_3d', 'industry_id_level2_pctChg_5d', 'industry_id_level2_pctChg_10d', 'industry_id_level2_pctChg_20d', 'industry_id_level2_pctChg_30d', 'industry_id_level2_pctChg_60d', 'industry_id_level2_pctChg_120d', 'industry_id_level2_pctChg_240d', 'industry_id_level2_pctChg_rank_ratio3d', 'industry_id_level2_pctChg_rank_ratio5d', 'industry_id_level2_pctChg_rank_ratio10d', 'industry_id_level2_pctChg_rank_ratio20d', 'industry_id_level2_pctChg_rank_ratio30d', 'industry_id_level2_pctChg_rank_ratio60d', 'industry_id_level2_pctChg_rank_ratio120d', 'industry_id_level2_pctChg_rank_ratio240d']
		industry2_feature = industry2_feature[industry2_col_name]
		industry2_feature['date'] = industry2_feature['date'].map(lambda x: int(x.replace('-', '')))

		feature = pd.merge(feature, industry2_feature, how="left", left_on=['date', "industry_id_level2"],right_on=['date', 'industry_id_level2'])
		del industry2_feature
		gc.collect()

		industry3_feature = pd.read_csv(self.industry3_fea_path)
		industry3_col_name = ['date', 'industry_id_level3', 'industry_id_level3_open_ratio', 'industry_id_level3_close_ratio', 'industry_id_level3_high_ratio', 'industry_id_level3_low_ratio', 'industry_id_level3_pctChg', 'industry_id_level3_rise_ratio', 'industry_id_level3_close', 'industry_id_level3_preclose', 'industry_id_level3_open', 'industry_id_level3_high', 'industry_id_level3_low', 'industry_id_level3_open_ratio_7d_avg', 'industry_id_level3_close_ratio_7d_avg', 'industry_id_level3_high_ratio_7d_avg', 'industry_id_level3_low_ratio_7d_avg', 'industry_id_level3_rsv_5', 'industry_id_level3_k_value_5', 'industry_id_level3_d_value_5', 'industry_id_level3_j_value_5', 'industry_id_level3_rsv_9', 'industry_id_level3_k_value_9', 'industry_id_level3_d_value_9', 'industry_id_level3_j_value_9', 'industry_id_level3_rsv_19', 'industry_id_level3_k_value_19', 'industry_id_level3_d_value_19', 'industry_id_level3_j_value_19', 'industry_id_level3_rsv_73', 'industry_id_level3_k_value_73', 'industry_id_level3_d_value_73', 'industry_id_level3_j_value_73', 'industry_id_level3_macd_dif_ratio', 'industry_id_level3_macd_positive_ratio_2', 'industry_id_level3_macd_positive_ratio_3', 'industry_id_level3_macd_positive_ratio_5', 'industry_id_level3_macd_positive_ratio_10', 'industry_id_level3_macd_positive_ratio_20', 'industry_id_level3_macd_positive_ratio_40', 'industry_id_level3_width_2', 'industry_id_level3_close_mb2_diff', 'industry_id_level3_width_3', 'industry_id_level3_close_mb3_diff', 'industry_id_level3_width_5', 'industry_id_level3_close_mb5_diff', 'industry_id_level3_width_10', 'industry_id_level3_close_mb10_diff', 'industry_id_level3_width_20', 'industry_id_level3_close_mb20_diff', 'industry_id_level3_width_40', 'industry_id_level3_close_mb40_diff', 'industry_id_level3_cr_bias_26d', 'industry_id_level3_rsi_3d', 'industry_id_level3_rsi_5d', 'industry_id_level3_rsi_10d', 'industry_id_level3_rsi_20d', 'industry_id_level3_rsi_40d', 'industry_id_level3_turn_3d_avg', 'industry_id_level3_turn_rank_3d_avg', 'industry_id_level3_turn_3dmax_dif', 'industry_id_level3_turn_3dmin_dif', 'industry_id_level3_turn_5d_avg', 'industry_id_level3_turn_rank_5d_avg', 'industry_id_level3_turn_5dmax_dif', 'industry_id_level3_turn_5dmin_dif', 'industry_id_level3_turn_10d_avg', 'industry_id_level3_turn_rank_10d_avg', 'industry_id_level3_turn_10dmax_dif', 'industry_id_level3_turn_10dmin_dif', 'industry_id_level3_turn_20d_avg', 'industry_id_level3_turn_rank_20d_avg', 'industry_id_level3_turn_20dmax_dif', 'industry_id_level3_turn_20dmin_dif', 'industry_id_level3_turn_30d_avg', 'industry_id_level3_turn_rank_30d_avg', 'industry_id_level3_turn_30dmax_dif', 'industry_id_level3_turn_30dmin_dif', 'industry_id_level3_turn_60d_avg', 'industry_id_level3_turn_rank_60d_avg', 'industry_id_level3_turn_60dmax_dif', 'industry_id_level3_turn_60dmin_dif', 'industry_id_level3_turn_120d_avg', 'industry_id_level3_turn_rank_120d_avg', 'industry_id_level3_turn_120dmax_dif', 'industry_id_level3_turn_120dmin_dif', 'industry_id_level3_turn_240d_avg', 'industry_id_level3_turn_rank_240d_avg', 'industry_id_level3_turn_240dmax_dif', 'industry_id_level3_turn_240dmin_dif', 'industry_id_level3_pctChg_3d', 'industry_id_level3_pctChg_5d', 'industry_id_level3_pctChg_10d', 'industry_id_level3_pctChg_20d', 'industry_id_level3_pctChg_30d', 'industry_id_level3_pctChg_60d', 'industry_id_level3_pctChg_120d', 'industry_id_level3_pctChg_240d', 'industry_id_level3_pctChg_rank_ratio3d', 'industry_id_level3_pctChg_rank_ratio5d', 'industry_id_level3_pctChg_rank_ratio10d', 'industry_id_level3_pctChg_rank_ratio20d', 'industry_id_level3_pctChg_rank_ratio30d', 'industry_id_level3_pctChg_rank_ratio60d', 'industry_id_level3_pctChg_rank_ratio120d', 'industry_id_level3_pctChg_rank_ratio240d']
		industry3_feature = industry3_feature[industry3_col_name]
		industry3_feature['date'] = industry3_feature['date'].map(lambda x: int(x.replace('-', '')))

		feature = pd.merge(feature, industry3_feature, how="left", left_on=['date', "industry_id_level3"],right_on=['date', 'industry_id_level3'])
		del industry3_feature
		gc.collect()

		# if self.year == 2008:
		# 	feature['date'] = pd.to_datetime(feature["date"], errors='coerce')
		feature = pd.merge(feature, label, how="inner", left_on=['date', "code"],right_on=['date', 'code'])
		del label
		gc.collect()
		feature = feature.round(6)
		feature.to_csv('{output_dir}/sample/{model_name}/train_sample_{year}.csv'.format(output_dir=self.output_dir, model_name='model_v18', year=str(self.year)), mode='w',header=True, index=False, encoding='utf-8')
		del feature
		gc.collect()
if __name__ == '__main__':
	base_path = 'E:/pythonProject/future/data/datafile'
	model_name = 'model_v12'
	years = [2008]
	years = [2008,2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
	for year in years:
		sample = Sample(year, base_path, model_name)
		sample.get_sample()
		# time.sleep(600)
