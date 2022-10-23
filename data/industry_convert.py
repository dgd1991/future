import pandas as pd
import numpy as np
industry_dict = np.load('E:/pythonProject/future/common/industry_dict.npy').item()
yearList = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
for year in yearList:
	file_name = 'E:/pythonProject/future/data/datafile/code_k_data_v2_' + year + '.csv'
	file = pd.read_csv(file_name)
	file['industry'] = file['industry'].map(lambda x: 0 if type(x) == type(np.nan) else industry_dict[x])
	file.to_csv(file_name, mode='w', header=True, index=False)

# file_name = 'E:/pythonProject/future/data/datafile/code_quarter_data_v2_all.csv'
# file = pd.read_csv(file_name)
# file['industry'] = file['industry'].map(lambda x: 0 if type(x) == type(np.nan) else industry_dict[x])
# file.to_csv(file_name, mode='w', header=True, index=False)
