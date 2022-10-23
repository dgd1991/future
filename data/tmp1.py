import csv
import gc
import multiprocessing

import baofuture as bs
import pandas as pd
import os
import copy

from feature.feature_process import *
import logging
from logging import handlers
#-*- coding: utf-8 -*-

import logging
from logging import handlers
import pandas as pd
import numpy as np

def get_industry_data(date):
    rs = bs.query_future_industry(date=date)
    industry_list = []
    while (rs.error_code == '0') & rs.next():
        industry_list.append(rs.get_row_data())
    industry_list = pd.DataFrame(industry_list)
    if (len(industry_list) == 0):
        print("=========================" + str(date) + "=========================")
        assert len(industry_list) == 0
    industry_list.columns = ['date', 'code', 'code_name', 'industry', 'industry_level']
    industry_list = industry_list[industry_list['industry'] != '']
    industry_list = industry_list[['code', 'industry']]
    return industry_list

_bs = bs
bs.login()
code_industry_data = get_industry_data('2012-01-19')
print(code_industry_data)