import gc
import os

import pandas as pd
month = '12'
day = '32'
years = [2013,2014,2015,2016]
index = 0
for year in years:
	year = str(year)
	index += 1
	file_name = 'E:/pythonProject/future/data/datafile/sample/model_v7/shuffled_train_sample_' + str(year) + '.csv'
	file = open("run.csv", "r+", encoding="utf-8")
	line = file.readline()

out_put = open('text02.txt','w')
out_put.write(line)