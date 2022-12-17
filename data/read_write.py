import gc
import os

import pandas as pd
month = '12'
day = '32'
years = [2013,2014]
index = 0
for year in years:
	year = str(year)
	index += 1
	file_name = 'E:/pythonProject/future/data/datafile/sample/model_v7/train_sample_' + str(year) + '.csv'
	file_name1 = 'E:/pythonProject/future/data/datafile/sample/model_v7/train_sample_2014_all' + '.csv'
	file = pd.read_csv(file_name)
	if index == 1:
		old = file
	else:
		old = pd.concat([old, file], axis=0)
	del file
	gc.collect()
	print(year)
old = old.sample(frac=1)
old.to_csv(file_name1, mode='w', header=True, index=False)
#

years = [2015,2016]
index = 0
for year in years:
	year = str(year)
	index += 1
	file_name = 'E:/pythonProject/future/data/datafile/sample/model_v7/train_sample_' + str(year) + '.csv'
	file_name1 = 'E:/pythonProject/future/data/datafile/sample/model_v7/train_sample_2016_all' + '.csv'
	file = pd.read_csv(file_name)
	if index == 1:
		old = file
	else:
		old = pd.concat([old, file], axis=0)
	del file
	gc.collect()
	print(year)
old = old.sample(frac=1)
old.to_csv(file_name1, mode='w', header=True, index=False)

years = [2017,2018]
index = 0
for year in years:
	year = str(year)
	index += 1
	file_name = 'E:/pythonProject/future/data/datafile/sample/model_v7/train_sample_' + str(year) + '.csv'
	file_name1 = 'E:/pythonProject/future/data/datafile/sample/model_v7/train_sample_2018_all' + '.csv'
	file = pd.read_csv(file_name)
	if index == 1:
		old = file
	else:
		old = pd.concat([old, file], axis=0)
	del file
	gc.collect()
	print(year)
old = old.sample(frac=1)
old.to_csv(file_name1, mode='w', header=True, index=False)

years = [2019,2020]
index = 0
for year in years:
	year = str(year)
	index += 1
	file_name = 'E:/pythonProject/future/data/datafile/sample/model_v7/train_sample_' + str(year) + '.csv'
	file_name1 = 'E:/pythonProject/future/data/datafile/sample/model_v7/train_sample_2020_all' + '.csv'
	file = pd.read_csv(file_name)
	if index == 1:
		old = file
	else:
		old = pd.concat([old, file], axis=0)
	del file
	gc.collect()
	print(year)
old = old.sample(frac=1)
old.to_csv(file_name1, mode='w', header=True, index=False)