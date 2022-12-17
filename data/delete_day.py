import gc
import os

import pandas as pd
month = '12'
day = '32'
year = 2022
date = '2022-12-12'
file_name = 'E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v5_' + str(year) + '.csv'
file = pd.read_csv(file_name)
file = file[file['date'] != date]
file.to_csv(file_name, mode='w', header=True, index=False)