import os
import pandas as pd
# years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

years = [2008]
for year in years:
	data = pd.read_csv('E:/pythonProject/future/data/datafile/sample/model_v12/train_sample_' + str(year) +'.csv')
	# float_fea_name = ['label_7','open_ratio','close_ratio','high_ratio','low_ratio','pctChg','open_ratio_7d_avg','close_ratio_7d_avg','high_ratio_7d_avg','low_ratio_7d_avg','rsv_3','k_value_3','d_value_3','j_value_3','rsv_5','k_value_5','d_value_5','j_value_5','rsv_9','k_value_9','d_value_9','j_value_9','rsv_19','k_value_19','d_value_19','j_value_19','rsv_73','k_value_73','d_value_73','j_value_73','macd_dif_ratio','macd_positive_ratio_2','macd_positive_ratio_3','macd_positive_ratio_5','macd_positive_ratio_10','macd_positive_ratio_20','macd_positive_ratio_40','width_2','close_mb2_diff','width_3','close_mb3_diff','width_5','close_mb5_diff','width_10','close_mb10_diff','width_20','close_mb20_diff','width_40','close_mb40_diff','cr_bias_26d','rsi_3d','rsi_5d','rsi_10d','rsi_20d','rsi_40d','turn_3d_avg','turn_rank_3d','turn_3dmax_dif','turn_3dmin_dif','turn_5d_avg','turn_rank_5d','turn_5dmax_dif','turn_5dmin_dif','turn_10d_avg','turn_rank_10d','turn_10dmax_dif','turn_10dmin_dif','turn_20d_avg','turn_rank_20d','turn_20dmax_dif','turn_20dmin_dif','turn_30d_avg','turn_rank_30d','turn_30dmax_dif','turn_30dmin_dif','turn_60d_avg','turn_rank_60d','turn_60dmax_dif','turn_60dmin_dif','turn_120d_avg','turn_rank_120d','turn_120dmax_dif','turn_120dmin_dif','turn_240d_avg','turn_rank_240d','turn_240dmax_dif','turn_240dmin_dif','pctChg_up_limit_3d','pctChg_down_limit_3d','pctChg_greater_3_3d','pctChg_greater_6_3d','pctChg_greater_9_3d','pctChg_greater_13_3d','pctChg_less_3_3d','pctChg_less_6_3d','pctChg_less_9_3d','pctChg_less_13_3d','turn_greater_3_3d','turn_greater_6_3d','turn_greater_10_3d','turn_greater_15_3d','turn_greater_21_3d','pctChg_up_limit_5d','pctChg_down_limit_5d','pctChg_greater_3_5d','pctChg_greater_6_5d','pctChg_greater_9_5d','pctChg_greater_13_5d','pctChg_less_3_5d','pctChg_less_6_5d','pctChg_less_9_5d','pctChg_less_13_5d','turn_greater_3_5d','turn_greater_6_5d','turn_greater_10_5d','turn_greater_15_5d','turn_greater_21_5d','pctChg_up_limit_10d','pctChg_down_limit_10d','pctChg_greater_3_10d','pctChg_greater_6_10d','pctChg_greater_9_10d','pctChg_greater_13_10d','pctChg_less_3_10d','pctChg_less_6_10d','pctChg_less_9_10d','pctChg_less_13_10d','turn_greater_3_10d','turn_greater_6_10d','turn_greater_10_10d','turn_greater_15_10d','turn_greater_21_10d','pctChg_up_limit_20d','pctChg_down_limit_20d','pctChg_greater_3_20d','pctChg_greater_6_20d','pctChg_greater_9_20d','pctChg_greater_13_20d','pctChg_less_3_20d','pctChg_less_6_20d','pctChg_less_9_20d','pctChg_less_13_20d','turn_greater_3_20d','turn_greater_6_20d','turn_greater_10_20d','turn_greater_15_20d','turn_greater_21_20d','pctChg_up_limit_30d','pctChg_down_limit_30d','pctChg_greater_3_30d','pctChg_greater_6_30d','pctChg_greater_9_30d','pctChg_greater_13_30d','pctChg_less_3_30d','pctChg_less_6_30d','pctChg_less_9_30d','pctChg_less_13_30d','turn_greater_3_30d','turn_greater_6_30d','turn_greater_10_30d','turn_greater_15_30d','turn_greater_21_30d','pctChg_up_limit_60d','pctChg_down_limit_60d','pctChg_greater_3_60d','pctChg_greater_6_60d','pctChg_greater_9_60d','pctChg_greater_13_60d','pctChg_less_3_60d','pctChg_less_6_60d','pctChg_less_9_60d','pctChg_less_13_60d','turn_greater_3_60d','turn_greater_6_60d','turn_greater_10_60d','turn_greater_15_60d','turn_greater_21_60d','pctChg_up_limit_120d','pctChg_down_limit_120d','pctChg_greater_3_120d','pctChg_greater_6_120d','pctChg_greater_9_120d','pctChg_greater_13_120d','pctChg_less_3_120d','pctChg_less_6_120d','pctChg_less_9_120d','pctChg_less_13_120d','turn_greater_3_120d','turn_greater_6_120d','turn_greater_10_120d','turn_greater_15_120d','turn_greater_21_120d','pctChg_3d','pctChg_5d','pctChg_10d','pctChg_20d','pctChg_30d','pctChg_60d','pctChg_120d','pctChg_rank_ratio3d','pctChg_rank_ratio_industry1_3d','pctChg_rank_ratio_industry2_3d','pctChg_rank_ratio_industry3_3d','turn_rank_industry1_3d','turn_rank_industry2_3d','turn_rank_industry3_3d','pctChg_rank_ratio5d','pctChg_rank_ratio_industry1_5d','pctChg_rank_ratio_industry2_5d','pctChg_rank_ratio_industry3_5d','turn_rank_industry1_5d','turn_rank_industry2_5d','turn_rank_industry3_5d','pctChg_rank_ratio10d','pctChg_rank_ratio_industry1_10d','pctChg_rank_ratio_industry2_10d','pctChg_rank_ratio_industry3_10d','turn_rank_industry1_10d','turn_rank_industry2_10d','turn_rank_industry3_10d','pctChg_rank_ratio20d','pctChg_rank_ratio_industry1_20d','pctChg_rank_ratio_industry2_20d','pctChg_rank_ratio_industry3_20d','turn_rank_industry1_20d','turn_rank_industry2_20d','turn_rank_industry3_20d','pctChg_rank_ratio30d','pctChg_rank_ratio_industry1_30d','pctChg_rank_ratio_industry2_30d','pctChg_rank_ratio_industry3_30d','turn_rank_industry1_30d','turn_rank_industry2_30d','turn_rank_industry3_30d','pctChg_rank_ratio60d','pctChg_rank_ratio_industry1_60d','pctChg_rank_ratio_industry2_60d','pctChg_rank_ratio_industry3_60d','turn_rank_industry1_60d','turn_rank_industry2_60d','turn_rank_industry3_60d','pctChg_rank_ratio120d','pctChg_rank_ratio_industry1_120d','pctChg_rank_ratio_industry2_120d','pctChg_rank_ratio_industry3_120d','turn_rank_industry1_120d','turn_rank_industry2_120d','turn_rank_industry3_120d']
	# float_fea = data[float_fea_name].groupby('label_7').mean()
	# if os.path.isfile('E:/pythonProject/future/data/datafile/sample/model_v12_eval/' + 'float_' +'.csv'):
	# 	float_fea.to_csv('E:/pythonProject/future/data/datafile/sample/model_v12_eval/' + 'float_' +'.csv', mode='a', header=True, index=True)
	# else:
	# 	float_fea.to_csv('E:/pythonProject/future/data/datafile/sample/model_v12_eval/' + 'float_' +'.csv', mode='w', header=True, index=True)

	bucket_fea_name = ['code_market','market_value','peTTM','pcfNcfTTM','pbMRQ','isST','turn','macd_positive','macd_dif_2','macd_dea_2','macd_2','macd_dif_3','macd_dea_3','macd_3','macd_dif_5','macd_dea_5','macd_5','macd_dif_10','macd_dea_10','macd_10','macd_dif_20','macd_dea_20','macd_20','macd_dif_40','macd_dea_40','macd_40','macd_dif_dea','cr_26d','cr_trend_26d','cr_trend_26d_0','cr_trend_26d_1','cr_trend_26d_2','cr_trend_26d_3','turn_3davg_dif','close_3davg_dif','close_3dmax_dif','close_3dmin_dif','close_3d_dif','turn_5davg_dif','turn_3_5d_avg','turn_3_5dmax_dif','turn_3_5dmin_dif','close_5davg_dif','close_5dmax_dif','close_5dmin_dif','close_5d_dif','close_3_5d_avg','turn_10davg_dif','turn_5_10d_avg','turn_5_10dmax_dif','turn_5_10dmin_dif','close_10davg_dif','close_10dmax_dif','close_10dmin_dif','close_10d_dif','close_5_10d_avg','turn_20davg_dif','turn_10_20d_avg','turn_10_20dmax_dif','turn_10_20dmin_dif','close_20davg_dif','close_20dmax_dif','close_20dmin_dif','close_20d_dif','close_10_20d_avg','turn_30davg_dif','turn_20_30d_avg','turn_20_30dmax_dif','turn_20_30dmin_dif','close_30davg_dif','close_30dmax_dif','close_30dmin_dif','close_30d_dif','close_20_30d_avg','turn_60davg_dif','turn_30_60d_avg','turn_30_60dmax_dif','turn_30_60dmin_dif','close_60davg_dif','close_60dmax_dif','close_60dmin_dif','close_60d_dif','close_30_60d_avg','turn_120davg_dif','turn_60_120d_avg','turn_60_120dmax_dif','turn_60_120dmin_dif','close_120davg_dif','close_120dmax_dif','close_120dmin_dif','close_120d_dif','close_60_120d_avg','turn_240davg_dif','turn_120_240d_avg','turn_120_240dmax_dif','turn_120_240dmin_dif','close_240davg_dif','close_240dmax_dif','close_240dmin_dif','close_240d_dif','close_120_240d_avg','max_turn_index3d','max_close_index3d','min_close_index3d','max_turn_close3d','max_closs_turn3d','min_closs_turn3d','max_turn_index5d','max_close_index5d','min_close_index5d','max_turn_close5d','max_closs_turn5d','min_closs_turn5d','max_turn_index10d','max_close_index10d','min_close_index10d','max_turn_close10d','max_closs_turn10d','min_closs_turn10d','max_turn_index20d','max_close_index20d','min_close_index20d','max_turn_close20d','max_closs_turn20d','min_closs_turn20d','max_turn_index30d','max_close_index30d','min_close_index30d','max_turn_close30d','max_closs_turn30d','min_closs_turn30d','max_turn_index60d','max_close_index60d','min_close_index60d','max_turn_close60d','max_closs_turn60d','min_closs_turn60d','max_turn_index120d','max_close_index120d','min_close_index120d','max_turn_close120d','max_closs_turn120d','min_closs_turn120d','max_turn_index240d','max_close_index240d','min_close_index240d','max_turn_close240d','max_closs_turn240d','min_closs_turn240d']
	# for fea in bucket_fea_name:
	# 	fea_tmp_list = [fea]
	# 	fea_tmp_list.append('label_7')
	# 	# fea_tmp_list['cnt'] = 1
	# 	bucket_fea = data[fea_tmp_list].groupby(fea_tmp_list).size().reset_index(name='cnt')
	# 	if os.path.isfile('E:/pythonProject/future/data/datafile/sample/model_v12_eval1/' + fea +'.csv'):
	# 		his_data = pd.read_csv('E:/pythonProject/future/data/datafile/sample/model_v12_eval1/' + fea + '.csv')
	# 		bucket_fea = pd.concat([his_data,bucket_fea],axis=0)
	# 		bucket_fea = bucket_fea.groupby(fea_tmp_list)['cnt'].sum().reset_index(name='cnt')
	# 		if year == 2022:
	# 			total = bucket_fea.groupby(fea)['cnt'].sum().reset_index(name='total_cnt')
	# 			bucket_fea = pd.merge(bucket_fea, total, how="left", left_on=[fea], right_on=[fea])
	# 			bucket_fea['ratio'] = bucket_fea['cnt']*1.0/bucket_fea['total_cnt']
	# 		bucket_fea.to_csv('E:/pythonProject/future/data/datafile/sample/model_v12_eval1/' + fea + '.csv', mode='w', header=True, index=False)
	# 	else:
	# 		bucket_fea.to_csv('E:/pythonProject/future/data/datafile/sample/model_v12_eval1/' + fea + '.csv', mode='w', header=True, index=False)

	# result_0 = []
	# result_1 = []
	# result_2 = []
	# result_3 = []
	# result_4 = []
	# result_5 = []
	# for fea in bucket_fea_name:
	# 	his_data = pd.read_csv('E:/pythonProject/future/data/datafile/sample/model_v12_eval1/' + fea + '.csv')
	# 	tmp = his_data[[fea, 'total_cnt']]
	# 	tmp.drop_duplicates(inplace=True)
	# 	tmp = tmp.sort_values(['total_cnt'], ascending=False)
	# 	list = tmp['total_cnt'].tolist()
	# 	all = sum(list)
	# 	new_list_value = [list[0]*1.0/all]
	# 	for i in range(1, len(list)):
	# 		tmp_value = list[i]*1.0/all
	# 		tmp_value += new_list_value[i-1]
	# 		new_list_value.append(tmp_value)
	# 	tmp['total_ratio'] = new_list_value
	# 	tmp = tmp[tmp['total_ratio']<0.6]
	# 	his_data = pd.merge(his_data, tmp, how="inner", left_on=[fea], right_on=[fea])
	# 	his_data = his_data[his_data['label_7']==1]
	# 	his_data['abs_ratio_diff'] = his_data['ratio'].map(lambda x: abs(x-0.1))
	# 	abs_value_avg_diff = his_data['abs_ratio_diff'].mean()
	# 	if abs_value_avg_diff<0.005:
	# 		result_0.append(fea)
	# 	if abs_value_avg_diff<0.01:
	# 		result_1.append(fea)
	# 	if abs_value_avg_diff<0.02:
	# 		result_2.append(fea)
	# 	if abs_value_avg_diff<0.03:
	# 		result_3.append(fea)
	# 	if abs_value_avg_diff<0.04:
	# 		result_4.append(fea)
	# 	if abs_value_avg_diff<0.05:
	# 		result_5.append(fea)
	# print(result_0)
	# print('\n')
	# print(result_1)
	# print('\n')
	# print(result_2)
	# print('\n')
	# print(result_3)
	# print('\n')
	# print(result_4)
	# print('\n')
	# print(result_5)
	# print('\n')

	result = []
	result_0 = []
	result_1 = []
	result_2 = []
	result_3 = []
	result_4 = []
	result_5 = []
	for fea in bucket_fea_name:
		his_data = pd.read_csv('E:/pythonProject/future/data/datafile/sample/model_v12_eval1/' + fea + '.csv')
		tmp = his_data[[fea, 'total_cnt']]
		tmp.drop_duplicates(inplace=True)
		tmp = tmp.sort_values(['total_cnt'], ascending=False)
		list = tmp['total_cnt'].tolist()
		all = sum(list)
		new_list_value = [i*1.0/all for i in list]
		# for i in range(1, len(list)):
		# 	tmp_value = list[i]*1.0/all
		# 	tmp_value += new_list_value[i-1]
		# 	new_list_value.append(tmp_value)
		tmp['total_ratio'] = new_list_value

		tmp = pd.concat([tmp, tmp], axis=0)
		tmp['label_7'] = [0]*len(list) + [1]*len(list)

		# tmp = tmp[tmp['total_ratio']<0.6]
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