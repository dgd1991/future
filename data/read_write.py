import pandas as pd
year = '2012'
month = '06'
day = '18'
file_name = 'E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v4_' + year + '.csv'
file = pd.read_csv(file_name)
file_out_put = file[file['date'] != year + '-' + month + '-' + day]
file_out_put.to_csv(file_name, mode='w', header=True, index=False)