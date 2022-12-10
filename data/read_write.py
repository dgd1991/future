import pandas as pd
# year = '2021'
month = '12'
day = '32'
years = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
for year in years:
	year = str(year)
	file_name = 'E:/pythonProject/future/data/datafile/sample/model_v6/train_sample_' + str(year) + '.csv'
	file_name1 = 'E:/pythonProject/future/data/datafile/sample/model_v7/train_sample_' + str(year) + '.csv'
	file1 = pd.read_csv(file_name1)
	file1 = file1[['date','code']]
	print(file1.shape)
	file1.drop_duplicates(inplace=True)
	file = pd.read_csv(file_name)
	file = pd.merge(file, file1, how="inner", left_on=['code', "date"],right_on=['code', 'date'])
	print(file.shape)
	file.to_csv(file_name, mode='w', header=True, index=False)