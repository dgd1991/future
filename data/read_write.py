import pandas as pd
# year = '2021'
month = '12'
day = '32'
years = [2022]
for year in years:
	year = str(year)
	file_name = 'E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v4_' + year + '.csv'
	file = pd.read_csv(file_name)
	file_out_put = file[file['date'] != year + '-' + month + '-' + day]
	file_out_put.drop_duplicates(inplace=True)
	file_out_put.to_csv(file_name, mode='w', header=True, index=False)