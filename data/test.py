import math

import numpy as np
import pandas as pd
# import tushare as ts
# pro=ts.pro_api('310e0e16357bb6357a57010274423a4393e67eb41ea3600db2b00391')
#
# his = pro.get_industry_classified(standard='sw')

# industry_raw_2014_path = 'E:/pythonProject/future/data/datafile/raw_feature/sw_industry_2014_raw.csv'
# industry_raw_2014 = pd.read_csv(industry_raw_2014_path, encoding='utf-8')
# industry_raw_2014.columns = ['industry_name_level1', 'industry_name_level2', 'industry_name_level3','industry_code']
# industry_raw_2014 = industry_raw_2014.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)
# industry_dic = {}
# for tup in industry_raw_2014.itertuples():
# 	# print(type(tup[1]))
# 	if type(tup[1]) == str:
# 		if industry_dic.__contains__(math.floor(tup[4]/1000)):
# 			print(industry_dic[math.floor(tup[4]/1000)])
# 			print(tup[1])
# 			print(tup[4])
# 			print(math.floor(tup[4]/1000))
# 		else:
# 			industry_dic[math.floor(tup[4] / 1000)] = tup[1]
# 	if type(tup[2]) == str:
# 		if industry_dic.__contains__(math.floor(tup[4]/100)):
# 			print(industry_dic[math.floor(tup[4]/100)])
# 			print(tup[1])
# 			print(tup[4])
# 			print(math.floor(tup[4]/100))
# 		else:
# 			industry_dic[math.floor(tup[4] / 100)] = tup[2]
# industry_raw_2014 = industry_raw_2014[~(industry_raw_2014['industry_name_level3'].isna())]
# industry_raw_2014['industry_name_level1'] = industry_raw_2014['industry_code'].map(lambda x: industry_dic[math.floor(x / 1000)])
# industry_raw_2014['industry_name_level2'] = industry_raw_2014['industry_code'].map(lambda x: industry_dic[math.floor(x / 100)])
# industry_raw_2014.to_csv('E:/pythonProject/future/data/datafile/raw_feature/sw_industry_all_2014.csv', mode='a', header=True, index=False)

# industry_raw_2014_path = 'E:/pythonProject/future/data/datafile/raw_feature/sw_industry_all_2014.csv'
# industry_raw_2014 = pd.read_csv(industry_raw_2014_path, encoding='utf-8')
# industry_raw_2014['industry_code'] = industry_raw_2014['industry_code'].map(lambda x: int(x))
# industry_raw_2014['year'] = 2014
# industry_raw_2021_path = 'E:/pythonProject/future/data/datafile/raw_feature/sw_industry_all_2021.csv'
# industry_raw_2021 = pd.read_csv(industry_raw_2021_path, encoding='utf-8')
# industry_raw_2021.columns = ['industry_code','industry_name_level1', 'industry_name_level2', 'industry_name_level3']
# industry_raw_2021 = industry_raw_2021[~(industry_raw_2021['industry_name_level3'].isna())]
# industry_raw_2021_final = industry_raw_2021[['industry_name_level1', 'industry_name_level2', 'industry_name_level3']]
# industry_raw_2021_final['industry_code'] = industry_raw_2021['industry_code']
# industry_raw_2021_final['year'] = 2021
# industry_raw_all = pd.concat([industry_raw_2014, industry_raw_2021_final], axis=0)
# industry_raw_all['row_num'] = industry_raw_all.groupby('industry_code')['year'].rank(ascending=False,method='first').astype(int)
# industry_raw_all = industry_raw_all.sort_values(by = ['industry_code','year'])
# print(industry_raw_all)
# industry_raw_all = industry_raw_all[industry_raw_all['row_num'] == 1]
# industry_raw_all_final = industry_raw_all[['industry_code','industry_name_level1', 'industry_name_level2', 'industry_name_level3']]
# industry_raw_all_final.to_csv('E:/pythonProject/future/data/datafile/raw_feature/sw_industry_all.csv', mode='a', header=True, index=False)


code_industry_path = 'E:/pythonProject/future/data/datafile/raw_feature/sw_code_industry.csv'
code_industry = pd.read_csv(code_industry_path, encoding='utf-8', converters={u'股票代码':str})
print(code_industry)
code_industry.columns = ['code', 'start_date', 'industry_code', 'update']
industry_all_path = 'E:/pythonProject/future/data/datafile/raw_feature/sw_industry_all.csv'
industry_all = pd.read_csv(industry_all_path, encoding='utf-8')
code_industry_all = pd.merge(code_industry, industry_all, how="inner", left_on=['industry_code'], right_on=['industry_code'])
# print(code_industry_all)
code_industry_all = code_industry_all.sort_values(by = ['code'])

from tools.Tools import Tools
tools = Tools()
tools.dictionary_load_path = 'E:/pythonProject/future/dictionary/sw_industry_dic.npy'
dictionary = tools.dictionary_load()
dictionary_level1 = dictionary[tools.industry_level1]
dictionary_level2 = dictionary[tools.industry_level2]
dictionary_level3 = dictionary[tools.industry_level3]
code_industry_all['industry_id_level1'] = code_industry_all['industry_name_level1'].map(lambda x: dictionary_level1[x])
code_industry_all['industry_id_level2'] = code_industry_all['industry_name_level2'].map(lambda x: dictionary_level2[x])
code_industry_all['industry_id_level3'] = code_industry_all['industry_name_level3'].map(lambda x: dictionary_level3[x])

code_industry_all['code'] = code_industry_all['code'].map(lambda x: tools.sw_code_to_bs_code(str(x)))
code_industry_all['start_date'] = code_industry_all['start_date'].map(lambda x: tools.sw_date_to_bs_date(x))

code_industry_all.to_csv('E:/pythonProject/future/data/datafile/raw_feature/sw_code_all_industry.csv', mode='a', header=True, index=False)




