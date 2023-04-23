import gc

import pandas as pd

pd.set_option('display.max_columns', 200)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)
#
# data1 = pd.read_csv('E:/pythonProject/future/data/datafile/label/2009_raw_v5.csv')
# data2 = pd.read_csv('E:/pythonProject/future/data/datafile/label/2009_raw_v4.csv')
# data3 = pd.merge(data2, data1, how="left", left_on=["date","code"], right_on=["date","code"])
# # print(data3.columns)
# data3 = data3[data3['pctChg_7_y'].isna()]
# print(data3)
# print(data.count())

data1 = pd.read_csv('E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_2007.csv')
data2 = pd.read_csv('E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_2008.csv')
data = pd.concat([data1, data2], axis=0)
raw_k_data = data[['date','code','pctChg',"industry_id_level1","industry_id_level2","industry_id_level3",'turn', 'close']]
raw_k_data = raw_k_data[(raw_k_data['industry_id_level3'] > 0) | (raw_k_data['code'] == 'sh.000001') | (raw_k_data['code'] == 'sz.399001') | (raw_k_data['code'] == 'sz.399006')]
raw_k_data = raw_k_data.groupby('code').apply(lambda x: x.set_index('date'))
# data=data[data['date']=='2008-01-03']
# data['rank'] = data['pctChg'].groupby(data['date']).rank(ascending=False)
# print(data)
# # data['rank1'] = data['pctChg'].groupby(data[['date',"industry_id_level1"]]).rank(ascending=False)
# # data['rank1'] = data[['pctChg','date',"industry_id_level1"]].groupby[['date',"industry_id_level1"]].Value.rank(ascending=False)
# # data = data.assign(rn=data.sort_values(['pctChg'], ascending=False).groupby(['date',"industry_id_level1"]).cumcount()+1)
# # data = data.assign(agge=data.sort_values(['pctChg'], ascending=False).groupby(['date',"industry_id_level1"]).cumcount()+1)
# data['industry_id_level1_rank'] = data.sort_values(['pctChg'], ascending=False).groupby(['date',"industry_id_level1"]).cumcount()+1
# data['industry_id_level2_rank'] = data.sort_values(['pctChg'], ascending=False).groupby(['date',"industry_id_level2"]).cumcount()+1
# data['industry_id_level3_rank'] = data.sort_values(['pctChg'], ascending=False).groupby(['date',"industry_id_level3"]).cumcount()+1
# print(data[['industry_id_level1','industry_id_level1_rank']].sort_values(by=['industry_id_level1','industry_id_level1_rank']))
# # print(data['industry_id_level1_rank'].min())
# # print(data['industry_id_level1_rank'].max())
# print(data['industry_id_level2_rank'].min())
# print(data['industry_id_level2_rank'].max())
# print(data['industry_id_level3_rank'].min())
# print(data['industry_id_level3_rank'].max())
# #
# tmp = data.groupby('industry_id_level1').size()
# print(tmp)
# # import pandas as pd
# data = pd.read_csv('E:/pythonProject/future/data/datafile/feature/model_v12/code_feature_2008.csv')
# data = data[data['date']<'2008-02-01']

# data[data['date']=='2008-01-03'][['industry_id_level1', 'pctChg_rank_ratio_industry1_120d']].groupby('industry_id_level1').min()
# data[data['industry_id_level1']==0][['industry_id_level1', 'industry_name_level1']]
#
#
# data[data['date']=='2008-01-03'].groupby('industry_id_level1').size()
tmp = raw_k_data[['turn', 'close']]
feature_all=raw_k_data[["industry_id_level1","industry_id_level2","industry_id_level3"]]
for day_cnt in [3, 5, 10, 20, 30, 60, 120]:
	feature_all['pctChg_' + str(day_cnt) + 'd'] = tmp['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=day_cnt, window=day_cnt, center=False).apply(lambda y: (y[day_cnt-1]-y[0])/y[0]))
feature_all = feature_all.reset_index(level=0, drop=False)
feature_all = feature_all.reset_index(level=0, drop=False)
feature_all = feature_all.sort_values(['date', 'code'])
feature_all = feature_all[feature_all['date'] > str(2008)]
del tmp
gc.collect()
tmp = feature_all[['date',"industry_id_level1","industry_id_level2","industry_id_level3"]]
for day_cnt in [3, 5, 10, 20, 30, 60, 120]:
	tmp['pctChg_' + str(day_cnt) + 'd'] = feature_all['pctChg_' + str(day_cnt) + 'd']
	tmp['pctChg_rank' + str(day_cnt) + 'd'] = tmp['pctChg_' + str(day_cnt) + 'd'].groupby(tmp['date']).rank(ascending=False)

	tmp['pctChg_rank_industry1_' + str(day_cnt) + 'd'] = tmp.sort_values(['pctChg_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level1']).cumcount()+1
	tmp['pctChg_rank_industry2_' + str(day_cnt) + 'd'] = tmp.sort_values(['pctChg_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level2']).cumcount()+1
	tmp['pctChg_rank_industry3_' + str(day_cnt) + 'd'] = tmp.sort_values(['pctChg_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level3']).cumcount()+1

	if day_cnt == 3:
		tmp['code_count'] = tmp.groupby('date')['pctChg_rank' + str(day_cnt) + 'd'].transform('max')
		tmp['industry1_count'] = tmp.groupby(['date','industry_id_level1'])['pctChg_rank_industry1_' + str(day_cnt) + 'd'].transform('max')
		tmp['industry2_count'] = tmp.groupby(['date','industry_id_level2'])['pctChg_rank_industry2_' + str(day_cnt) + 'd'].transform('max')
		tmp['industry3_count'] = tmp.groupby(['date','industry_id_level3'])['pctChg_rank_industry3_' + str(day_cnt) + 'd'].transform('max')

	feature_all['pctChg_rank_ratio' + str(day_cnt) + 'd'] = (tmp['pctChg_rank' + str(day_cnt) + 'd']/tmp['code_count'])
	feature_all['pctChg_rank_ratio_industry1_' + str(day_cnt) + 'd'] = (tmp['pctChg_rank_industry1_' + str(day_cnt) + 'd']/tmp['industry1_count'])
	feature_all['pctChg_rank_ratio_industry2_' + str(day_cnt) + 'd'] = (tmp['pctChg_rank_industry2_' + str(day_cnt) + 'd']/tmp['industry2_count'])
	feature_all['pctChg_rank_ratio_industry3_' + str(day_cnt) + 'd'] = (tmp['pctChg_rank_industry3_' + str(day_cnt) + 'd']/tmp['industry3_count'])

	tmp['turn_rank_' + str(day_cnt) + 'd'] = feature_all['turn_rank_' + str(day_cnt) + 'd']
	tmp['turn_rank_' + str(day_cnt) + 'd'] = tmp['turn_rank_' + str(day_cnt) + 'd'].groupby(tmp['date']).rank(ascending=False)

	tmp['turn_rank_industry1_' + str(day_cnt) + 'd'] = tmp.sort_values(['turn_rank_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level1']).cumcount()+1
	tmp['turn_rank_industry2_' + str(day_cnt) + 'd'] = tmp.sort_values(['turn_rank_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level2']).cumcount()+1
	tmp['turn_rank_industry3_' + str(day_cnt) + 'd'] = tmp.sort_values(['turn_rank_' + str(day_cnt) + 'd'], ascending=False).groupby(['date','industry_id_level3']).cumcount()+1


	feature_all['turn_rank_' + str(day_cnt) + 'd'] = (tmp['turn_rank_' + str(day_cnt) + 'd']/tmp['code_count'])
	feature_all['turn_rank_industry1_' + str(day_cnt) + 'd'] = (tmp['turn_rank_industry1_' + str(day_cnt) + 'd']/tmp['code_count'])
	feature_all['turn_rank_industry2_' + str(day_cnt) + 'd'] = (tmp['turn_rank_industry2_' + str(day_cnt) + 'd']/tmp['code_count'])
	feature_all['turn_rank_industry3_' + str(day_cnt) + 'd'] = (tmp['turn_rank_industry3_' + str(day_cnt) + 'd']/tmp['code_count'])

feature_all = feature_all.reset_index(level=0, drop=True)

tmp[(tmp['date']=='2008-01-03') & (tmp['industry_id_level1']==1.0)][['industry_id_level1','pctChg_rank_industry1_3d']].sort_values(['pctChg_rank_industry1_3d'])


tmp[(tmp['date']=='2008-01-03') & (tmp['industry_id_level1']==1.0)][['industry1_count']]

data1 = pd.read_csv('E:/pythonProject/future/data/datafile/sample/model_v11/train_sample_2019.csv')
data1['label_7'].value_counts()
data1 = pd.read_csv('E:/pythonProject/future/data/datafile/sample/model_v12/train_sample_2015.csv')
