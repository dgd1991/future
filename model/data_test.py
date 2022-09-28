import baostock as bs
import pandas as pd
import os
import copy
from feature.feature_process import *

bs.login()
date='2022-06-17'
# rs = bs.query_stock_industry(date="2006-01-02")
# industry_list = []
rs_profit = bs.query_profit_data(code="sh.600367", year=2007, quarter=1)
while (rs_profit.error_code == '0') & rs_profit.next():
    print(rs_profit.get_row_data())

# industry_list.columns = ['date', 'code', 'code_name', 'industry', 'industry_level']
# industry_list = industry_list[(industry_list['industry'] != '') | (industry_list['industry'] == '银行')]
# industry_list = industry_list[['code', 'industry']]
# print(industry_list)
# print(len(industry_list))
# print(industry_list[industry_list['industry'] != ''].count())