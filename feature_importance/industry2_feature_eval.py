import os
import pandas as pd
years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
float_fea_name = ['industry_id_level2_pctChg_rank_ratio240d', 'industry_id_level2_pctChg_240d', 'industry_id_level2_open_ratio', 'industry_id_level2_close_ratio', 'industry_id_level2_high_ratio', 'industry_id_level2_low_ratio', 'industry_id_level2_pctChg', 'industry_id_level2_rise_ratio', 'industry_id_level2_open_ratio_7d_avg', 'industry_id_level2_close_ratio_7d_avg', 'industry_id_level2_high_ratio_7d_avg', 'industry_id_level2_low_ratio_7d_avg', 'industry_id_level2_rsv_5', 'industry_id_level2_k_value_5', 'industry_id_level2_d_value_5', 'industry_id_level2_j_value_5', 'industry_id_level2_rsv_9', 'industry_id_level2_k_value_9', 'industry_id_level2_d_value_9', 'industry_id_level2_j_value_9', 'industry_id_level2_rsv_19', 'industry_id_level2_k_value_19', 'industry_id_level2_d_value_19', 'industry_id_level2_j_value_19', 'industry_id_level2_rsv_73', 'industry_id_level2_k_value_73', 'industry_id_level2_d_value_73', 'industry_id_level2_j_value_73', 'industry_id_level2_macd_dif_ratio', 'industry_id_level2_macd_positive_ratio_2', 'industry_id_level2_macd_positive_ratio_3', 'industry_id_level2_macd_positive_ratio_5', 'industry_id_level2_macd_positive_ratio_10', 'industry_id_level2_macd_positive_ratio_20', 'industry_id_level2_macd_positive_ratio_40', 'industry_id_level2_width_2', 'industry_id_level2_close_mb2_diff', 'industry_id_level2_width_3', 'industry_id_level2_close_mb3_diff', 'industry_id_level2_width_5', 'industry_id_level2_close_mb5_diff', 'industry_id_level2_width_10', 'industry_id_level2_close_mb10_diff', 'industry_id_level2_width_20', 'industry_id_level2_close_mb20_diff', 'industry_id_level2_width_40', 'industry_id_level2_close_mb40_diff', 'industry_id_level2_cr_bias_26d', 'industry_id_level2_rsi_3d', 'industry_id_level2_rsi_5d', 'industry_id_level2_rsi_10d', 'industry_id_level2_rsi_20d', 'industry_id_level2_rsi_40d', 'industry_id_level2_turn_3d_avg', 'industry_id_level2_turn_rank_3d_avg', 'industry_id_level2_turn_3dmax_dif', 'industry_id_level2_turn_3dmin_dif', 'industry_id_level2_turn_5d_avg', 'industry_id_level2_turn_rank_5d_avg', 'industry_id_level2_turn_5dmax_dif', 'industry_id_level2_turn_5dmin_dif', 'industry_id_level2_turn_10d_avg', 'industry_id_level2_turn_rank_10d_avg', 'industry_id_level2_turn_10dmax_dif', 'industry_id_level2_turn_10dmin_dif', 'industry_id_level2_turn_20d_avg', 'industry_id_level2_turn_rank_20d_avg', 'industry_id_level2_turn_20dmax_dif', 'industry_id_level2_turn_20dmin_dif', 'industry_id_level2_turn_30d_avg', 'industry_id_level2_turn_rank_30d_avg', 'industry_id_level2_turn_30dmax_dif', 'industry_id_level2_turn_30dmin_dif', 'industry_id_level2_turn_60d_avg', 'industry_id_level2_turn_rank_60d_avg', 'industry_id_level2_turn_60dmax_dif', 'industry_id_level2_turn_60dmin_dif', 'industry_id_level2_turn_120d_avg', 'industry_id_level2_turn_rank_120d_avg', 'industry_id_level2_turn_120dmax_dif', 'industry_id_level2_turn_120dmin_dif', 'industry_id_level2_turn_240d_avg', 'industry_id_level2_turn_rank_240d_avg', 'industry_id_level2_turn_240dmax_dif', 'industry_id_level2_turn_240dmin_dif', 'industry_id_level2_pctChg_3d', 'industry_id_level2_pctChg_5d', 'industry_id_level2_pctChg_10d', 'industry_id_level2_pctChg_20d', 'industry_id_level2_pctChg_30d', 'industry_id_level2_pctChg_60d', 'industry_id_level2_pctChg_120d', 'industry_id_level2_pctChg_rank_ratio3d', 'industry_id_level2_pctChg_rank_ratio5d', 'industry_id_level2_pctChg_rank_ratio10d', 'industry_id_level2_pctChg_rank_ratio20d', 'industry_id_level2_pctChg_rank_ratio30d', 'industry_id_level2_pctChg_rank_ratio60d', 'industry_id_level2_pctChg_rank_ratio120d', 'label_7', 'pctChg_7', 'sh_pctChg_7', 'label_15', 'pctChg_15', 'sh_pctChg_15']
bucket_fea_name = ['industry_id_level2_turn', 'industry_id_level2_peTTM', 'industry_id_level2_pcfNcfTTM', 'industry_id_level2_pbMRQ', 'industry_id_level2_market_value', 'industry_id_level2_pctChg_up_limit', 'industry_id_level2_pctChg_down_limit', 'industry_id_level2_pctChg_up_limit_3', 'industry_id_level2_pctChg_down_limit_3', 'industry_id_level2_pctChg_up_limit_7', 'industry_id_level2_pctChg_down_limit_7', 'industry_id_level2_pctChg_up_limit_15', 'industry_id_level2_pctChg_down_limit_15', 'industry_id_level2_pctChg_up_limit_30', 'industry_id_level2_pctChg_down_limit_30', 'industry_id_level2_macd_positive', 'industry_id_level2_macd_dif_2', 'industry_id_level2_macd_dea_2', 'industry_id_level2_macd_2', 'industry_id_level2_macd_dif_3', 'industry_id_level2_macd_dea_3', 'industry_id_level2_macd_3', 'industry_id_level2_macd_dif_5', 'industry_id_level2_macd_dea_5', 'industry_id_level2_macd_5', 'industry_id_level2_macd_dif_10', 'industry_id_level2_macd_dea_10', 'industry_id_level2_macd_10', 'industry_id_level2_macd_dif_20', 'industry_id_level2_macd_dea_20', 'industry_id_level2_macd_20', 'industry_id_level2_macd_dif_40', 'industry_id_level2_macd_dea_40', 'industry_id_level2_macd_40', 'industry_id_level2_macd_dif_dea', 'industry_id_level2_cr_26d', 'industry_id_level2_cr_trend_26d', 'industry_id_level2_cr_trend_26d_0', 'industry_id_level2_cr_trend_26d_1', 'industry_id_level2_cr_trend_26d_2', 'industry_id_level2_cr_trend_26d_3', 'industry_id_level2_turn_3davg_dif', 'industry_id_level2_close_3davg_dif', 'industry_id_level2_close_3dmax_dif', 'industry_id_level2_close_3dmin_dif', 'industry_id_level2_close_3d_dif', 'industry_id_level2_turn_5davg_dif', 'industry_id_level2_turn_3_5d_avg', 'industry_id_level2_turn_3_5dmax_dif', 'industry_id_level2_turn_3_5dmin_dif', 'industry_id_level2_close_5davg_dif', 'industry_id_level2_close_5dmax_dif', 'industry_id_level2_close_5dmin_dif', 'industry_id_level2_close_5d_dif', 'industry_id_level2_close_3_5d_avg', 'industry_id_level2_turn_10davg_dif', 'industry_id_level2_turn_5_10d_avg', 'industry_id_level2_turn_5_10dmax_dif', 'industry_id_level2_turn_5_10dmin_dif', 'industry_id_level2_close_10davg_dif', 'industry_id_level2_close_10dmax_dif', 'industry_id_level2_close_10dmin_dif', 'industry_id_level2_close_10d_dif', 'industry_id_level2_close_5_10d_avg', 'industry_id_level2_turn_20davg_dif', 'industry_id_level2_turn_10_20d_avg', 'industry_id_level2_turn_10_20dmax_dif', 'industry_id_level2_turn_10_20dmin_dif', 'industry_id_level2_close_20davg_dif', 'industry_id_level2_close_20dmax_dif', 'industry_id_level2_close_20dmin_dif', 'industry_id_level2_close_20d_dif', 'industry_id_level2_close_10_20d_avg', 'industry_id_level2_turn_30davg_dif', 'industry_id_level2_turn_20_30d_avg', 'industry_id_level2_turn_20_30dmax_dif', 'industry_id_level2_turn_20_30dmin_dif', 'industry_id_level2_close_30davg_dif', 'industry_id_level2_close_30dmax_dif', 'industry_id_level2_close_30dmin_dif', 'industry_id_level2_close_30d_dif', 'industry_id_level2_close_20_30d_avg', 'industry_id_level2_turn_60davg_dif', 'industry_id_level2_turn_30_60d_avg', 'industry_id_level2_turn_30_60dmax_dif', 'industry_id_level2_turn_30_60dmin_dif', 'industry_id_level2_close_60davg_dif', 'industry_id_level2_close_60dmax_dif', 'industry_id_level2_close_60dmin_dif', 'industry_id_level2_close_60d_dif', 'industry_id_level2_close_30_60d_avg', 'industry_id_level2_turn_120davg_dif', 'industry_id_level2_turn_60_120d_avg', 'industry_id_level2_turn_60_120dmax_dif', 'industry_id_level2_turn_60_120dmin_dif', 'industry_id_level2_close_120davg_dif', 'industry_id_level2_close_120dmax_dif', 'industry_id_level2_close_120dmin_dif', 'industry_id_level2_close_120d_dif', 'industry_id_level2_close_60_120d_avg', 'industry_id_level2_turn_240davg_dif', 'industry_id_level2_turn_120_240d_avg', 'industry_id_level2_turn_120_240dmax_dif', 'industry_id_level2_turn_120_240dmin_dif', 'industry_id_level2_close_240davg_dif', 'industry_id_level2_close_240dmax_dif', 'industry_id_level2_close_240dmin_dif', 'industry_id_level2_close_240d_dif', 'industry_id_level2_close_120_240d_avg', 'industry_id_level2_max_turn_index3d', 'industry_id_level2_max_close_index3d', 'industry_id_level2_min_close_index3d', 'industry_id_level2_max_turn_close3d', 'industry_id_level2_max_closs_turn3d', 'industry_id_level2_min_closs_turn3d', 'industry_id_level2_max_turn_index5d', 'industry_id_level2_max_close_index5d', 'industry_id_level2_min_close_index5d', 'industry_id_level2_max_turn_close5d', 'industry_id_level2_max_closs_turn5d', 'industry_id_level2_min_closs_turn5d', 'industry_id_level2_max_turn_index10d', 'industry_id_level2_max_close_index10d', 'industry_id_level2_min_close_index10d', 'industry_id_level2_max_turn_close10d', 'industry_id_level2_max_closs_turn10d', 'industry_id_level2_min_closs_turn10d', 'industry_id_level2_max_turn_index20d', 'industry_id_level2_max_close_index20d', 'industry_id_level2_min_close_index20d', 'industry_id_level2_max_turn_close20d', 'industry_id_level2_max_closs_turn20d', 'industry_id_level2_min_closs_turn20d', 'industry_id_level2_max_turn_index30d', 'industry_id_level2_max_close_index30d', 'industry_id_level2_min_close_index30d', 'industry_id_level2_max_turn_close30d', 'industry_id_level2_max_closs_turn30d', 'industry_id_level2_min_closs_turn30d', 'industry_id_level2_max_turn_index60d', 'industry_id_level2_max_close_index60d', 'industry_id_level2_min_close_index60d', 'industry_id_level2_max_turn_close60d', 'industry_id_level2_max_closs_turn60d', 'industry_id_level2_min_closs_turn60d', 'industry_id_level2_max_turn_index120d', 'industry_id_level2_max_close_index120d', 'industry_id_level2_min_close_index120d', 'industry_id_level2_max_turn_close120d', 'industry_id_level2_max_closs_turn120d', 'industry_id_level2_min_closs_turn120d', 'industry_id_level2_max_turn_index240d', 'industry_id_level2_max_close_index240d', 'industry_id_level2_min_close_index240d', 'industry_id_level2_max_turn_close240d', 'industry_id_level2_max_closs_turn240d', 'industry_id_level2_min_closs_turn240d']
#
for year in years:
	data = pd.read_csv('E:/pythonProject/future/data/datafile/sample/model_v14/train_sample_' + str(year) +'.csv')
	float_fea = data[float_fea_name].groupby('label_7').mean()
	if os.path.isfile('E:/pythonProject/future/data/datafile/sample/model_v14_eval/' + 'float_' +'.csv'):
		float_fea.to_csv('E:/pythonProject/future/data/datafile/sample/model_v14_eval/' + 'float_' +'.csv', mode='a', header=True, index=True)
	else:
		float_fea.to_csv('E:/pythonProject/future/data/datafile/sample/model_v14_eval/' + 'float_' +'.csv', mode='w', header=True, index=True)

	for fea in bucket_fea_name:
		fea_tmp_list = [fea]
		fea_tmp_list.append('label_7')
		# fea_tmp_list['cnt'] = 1
		bucket_fea = data[fea_tmp_list].groupby(fea_tmp_list).size().reset_index(name='cnt')
		if os.path.isfile('E:/pythonProject/future/data/datafile/sample/model_v14_eval/' + fea +'.csv'):
			his_data = pd.read_csv('E:/pythonProject/future/data/datafile/sample/model_v14_eval/' + fea + '.csv')
			bucket_fea = pd.concat([his_data,bucket_fea],axis=0)
			bucket_fea = bucket_fea.groupby(fea_tmp_list)['cnt'].sum().reset_index(name='cnt')
			if year == 2022:
				total = bucket_fea.groupby(fea)['cnt'].sum().reset_index(name='total_cnt')
				bucket_fea = pd.merge(bucket_fea, total, how="left", left_on=[fea], right_on=[fea])
				bucket_fea['ratio'] = bucket_fea['cnt']*1.0/bucket_fea['total_cnt']
			bucket_fea.to_csv('E:/pythonProject/future/data/datafile/sample/model_v14_eval/' + fea + '.csv', mode='w', header=True, index=False)
		else:
			bucket_fea.to_csv('E:/pythonProject/future/data/datafile/sample/model_v14_eval/' + fea + '.csv', mode='w', header=True, index=False)

result = []
result_0 = []
result_1 = []
result_2 = []
result_3 = []
result_4 = []
result_5 = []
for fea in bucket_fea_name:
	his_data = pd.read_csv('E:/pythonProject/future/data/datafile/sample/model_v14_eval/' + fea + '.csv')
	tmp = his_data[[fea, 'total_cnt']]
	tmp.drop_duplicates(inplace=True)
	tmp = tmp.sort_values(['total_cnt'], ascending=False)
	list = tmp['total_cnt'].tolist()
	all = sum(list)
	new_list_value = [i*1.0/all for i in list]

	tmp['total_ratio'] = new_list_value

	tmp = pd.concat([tmp, tmp], axis=0)
	tmp['label_7'] = [0]*len(list) + [1]*len(list)

	his_data = pd.merge(tmp, his_data, how="left", left_on=[fea, 'label_7'], right_on=[fea, 'label_7'])
	his_data = his_data[his_data['label_7']==1]
	his_data['abs_ratio_diff'] = his_data.apply(lambda x: abs(x.ratio-0.1)*x.total_ratio if x.ratio>0 else 0.1*x.total_ratio, axis=1)
	abs_value_avg_diff = his_data['abs_ratio_diff'].sum()
	if abs_value_avg_diff<0.0025:
		result.append(fea)
	if abs_value_avg_diff<0.005:
		result_0.append(fea)
	if abs_value_avg_diff<0.01:
		result_1.append(fea)
	if abs_value_avg_diff<0.02:
		result_2.append(fea)
	if abs_value_avg_diff<0.03:
		result_3.append(fea)
	if abs_value_avg_diff<0.04:
		result_4.append(fea)
	if abs_value_avg_diff<0.05:
		result_5.append(fea)
print(result)
print('\n')
print(result_0)
print('\n')
print(result_1)
print('\n')
print(result_2)
print('\n')
print(result_3)
print('\n')
print(result_4)
print('\n')
print(result_5)
print('\n')