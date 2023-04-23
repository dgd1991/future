import pandas as pd
# data1 = pd.read_csv('E:/pythonProject/future/data/datafile/sample/model_v14/train_sample_2015.csv')
# print(data1['label_7'].value_counts())
# data2 = pd.read_csv('E:/pythonProject/future/data/datafile/sample/model_v14/train_sample_2015.csv')
# print(data2['label_7'].value_counts())
#
# col = ['industry_id_level2_open_ratio','industry_id_level2_open_ratio_7d_avg','industry_id_level2_macd_positive_ratio_20','industry_id_level2_turn_60d_avg','industry_id_level2_turn_60dmax_dif','pctChg_rank_ratio120d','industry_id_level2_macd_positive_ratio_40','industry_id_level2_close_mb2_diff','industry_id_level2_close_mb3_diff','industry_id_level2_rsi_3d','industry_id_level2_pctChg_3d','industry_id_level2_low_ratio','industry_id_level2_j_value_5','industry_id_level2_macd_positive_ratio_10','industry_id_level2_cr_bias_26d','industry_id_level2_turn_3d_avg','industry_id_level2_turn_rank_3d_avg','industry_id_level2_turn_3dmax_dif','industry_id_level2_turn_3dmin_dif','industry_id_level2_turn_5d_avg','industry_id_level2_turn_rank_5d_avg','industry_id_level2_turn_5dmax_dif','industry_id_level2_turn_5dmin_dif','industry_id_level2_turn_10d_avg','industry_id_level2_turn_rank_10d_avg','industry_id_level2_turn_10dmax_dif','industry_id_level2_turn_10dmin_dif','industry_id_level2_turn_20d_avg','industry_id_level2_turn_rank_20d_avg','industry_id_level2_turn_20dmax_dif','industry_id_level2_turn_20dmin_dif','industry_id_level2_turn_30d_avg','industry_id_level2_turn_rank_30d_avg','industry_id_level2_turn_30dmax_dif','industry_id_level2_turn_30dmin_dif','industry_id_level2_turn_120d_avg','industry_id_level2_turn_120dmax_dif','industry_id_level2_turn_240d_avg','industry_id_level2_turn_rank_240d_avg','industry_id_level2_turn_240dmax_dif','industry_id_level2_pctChg_5d','industry_id_level2_pctChg_120d','pctChg_rank_ratio3d','industry_id_level2_close_ratio','industry_id_level2_high_ratio','industry_id_level2_pctChg','industry_id_level2_rise_ratio','industry_id_level2_close_ratio_7d_avg','industry_id_level2_high_ratio_7d_avg','industry_id_level2_low_ratio_7d_avg','industry_id_level2_rsv_5','industry_id_level2_k_value_5','industry_id_level2_d_value_5','industry_id_level2_rsv_9','industry_id_level2_k_value_9','industry_id_level2_d_value_9','industry_id_level2_j_value_9','industry_id_level2_rsv_19','industry_id_level2_k_value_19','industry_id_level2_d_value_19','industry_id_level2_j_value_19','industry_id_level2_rsv_73','industry_id_level2_k_value_73','industry_id_level2_d_value_73','industry_id_level2_j_value_73','industry_id_level2_macd_dif_ratio','industry_id_level2_macd_positive_ratio_2','industry_id_level2_macd_positive_ratio_3','industry_id_level2_macd_positive_ratio_5','industry_id_level2_width_2','industry_id_level2_width_3','industry_id_level2_width_5','industry_id_level2_close_mb5_diff','industry_id_level2_width_10','industry_id_level2_close_mb10_diff','industry_id_level2_width_20','industry_id_level2_close_mb20_diff','industry_id_level2_width_40','industry_id_level2_close_mb40_diff','industry_id_level2_rsi_5d','industry_id_level2_rsi_10d','industry_id_level2_rsi_20d','industry_id_level2_rsi_40d','industry_id_level2_turn_rank_60d_avg','industry_id_level2_turn_60dmin_dif','industry_id_level2_turn_rank_120d_avg','industry_id_level2_turn_120dmin_dif','industry_id_level2_turn_240dmin_dif','industry_id_level2_pctChg_10d','industry_id_level2_pctChg_20d','industry_id_level2_pctChg_30d','industry_id_level2_pctChg_60d','pctChg_rank_ratio5d','pctChg_rank_ratio10d','pctChg_rank_ratio20d','pctChg_rank_ratio30d','pctChg_rank_ratio60d']
# data2 = pd.read_csv('E:/pythonProject/future/data/datafile/feature/model_v12/industry2_feature_2015.csv')
# data2=data2[col]
# a=data2.min()
# b = data2.max()
# a.to_csv('E:/pythonProject/future/data/datafile/sample/model_v14/min_new.csv', mode='w', header=False, index=True, encoding='utf-8')
# b.to_csv('E:/pythonProject/future/data/datafile/sample/model_v14/max_new.csv', mode='w', header=False, index=True, encoding='utf-8')




col = ['turn_rank_240d','turn_rank_120d','turn_rank_60d']
raw_k_data = pd.read_csv('E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_2008.csv')
tmp = raw_k_data[['code','date', 'pctChg', 'amount', 'turn']]
tmp = tmp.sort_values(['date', 'code'])

tmp['pctChg_rank'] = tmp.sort_values(['pctChg'], ascending=False).groupby(['date']).cumcount()+1
tmp['amount_rank'] = tmp.sort_values(['amount'], ascending=False).groupby(['date']).cumcount() + 1
tmp['turn_rank'] = tmp.sort_values(['turn'], ascending=False).groupby(['date']).cumcount() + 1
tmp['code_count'] = tmp.groupby('date')['pctChg_rank'].transform('max')
tmp['pctChg_rank_ratio'] = tmp['pctChg_rank']/tmp['code_count']
tmp['amount_rank_ratio'] = tmp['amount_rank'] / tmp['code_count']
tmp['turn_rank_ratio'] = tmp['turn_rank'] / tmp['code_count']
tmp['pctChg_top1'] = tmp['pctChg_rank_ratio'].map(lambda x: 1 if x<0.01 else 0)
tmp['pctChg_top5'] = tmp['pctChg_rank_ratio'].map(lambda x: 1 if x < 0.05 else 0)
tmp['pctChg_top10'] = tmp['pctChg_rank_ratio'].map(lambda x: 1 if x < 0.1 else 0)
tmp['pctChg_top20'] = tmp['pctChg_rank_ratio'].map(lambda x: 1 if x < 0.2 else 0)
tmp['pctChg_top30'] = tmp['pctChg_rank_ratio'].map(lambda x: 1 if x < 0.3 else 0)

tmp['amount_top1'] = tmp['amount_rank_ratio'].map(lambda x: 1 if x<0.01 else 0)
tmp['amount_top5'] = tmp['amount_rank_ratio'].map(lambda x: 1 if x < 0.05 else 0)
tmp['amount_top10'] = tmp['amount_rank_ratio'].map(lambda x: 1 if x < 0.1 else 0)

tmp['turn_top1'] = tmp['turn_rank_ratio'].map(lambda x: 1 if x<0.01 else 0)
tmp['turn_top5'] = tmp['turn_rank_ratio'].map(lambda x: 1 if x < 0.05 else 0)
tmp['turn_top10'] = tmp['turn_rank_ratio'].map(lambda x: 1 if x < 0.1 else 0)
tmp['turn_top20'] = tmp['turn_rank_ratio'].map(lambda x: 1 if x < 0.2 else 0)
tmp['turn_top30'] = tmp['turn_rank_ratio'].map(lambda x: 1 if x < 0.3 else 0)
topN_percent_ratio = tmp[['date', 'code']]
for day_cnt in [1, 3, 5, 8, 15, 30, 60, 120, 240]:
	topN_percent_ratio['pctChg_top1_' + str(day_cnt) + 'd'] = tmp[['pctChg_top1','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
	topN_percent_ratio['pctChg_top5_' + str(day_cnt) + 'd'] = tmp[['pctChg_top5','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
	topN_percent_ratio['pctChg_top10_' + str(day_cnt) + 'd'] = tmp[['pctChg_top10','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
	topN_percent_ratio['pctChg_top20_' + str(day_cnt) + 'd'] = tmp[['pctChg_top20','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
	topN_percent_ratio['pctChg_top30_' + str(day_cnt) + 'd'] = tmp[['pctChg_top30','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())

	topN_percent_ratio['amount_top1_' + str(day_cnt) + 'd'] = tmp[['amount_top1','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
	topN_percent_ratio['amount_top5_' + str(day_cnt) + 'd'] = tmp[['amount_top5','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
	topN_percent_ratio['amount_top10_' + str(day_cnt) + 'd'] = tmp[['amount_top10','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())

	topN_percent_ratio['turn_top1_' + str(day_cnt) + 'd'] = tmp[['turn_top1','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
	topN_percent_ratio['turn_top5_' + str(day_cnt) + 'd'] = tmp[['turn_top5','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
	topN_percent_ratio['turn_top10_' + str(day_cnt) + 'd'] = tmp[['turn_top10','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
	topN_percent_ratio['turn_top20_' + str(day_cnt) + 'd'] = tmp[['turn_top20','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())
	topN_percent_ratio['turn_top30_' + str(day_cnt) + 'd'] = tmp[['turn_top30','code']].groupby(['code']).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).mean())

# topN_percent_ratio = topN_percent_ratio[topN_percent_ratio['date'] > str(self.year)]