import datetime
import hashlib
import math
from calendar import calendar

import numpy as np
import pandas as pd


class Tools(object):
    def __init__(self):
        self.industry_all_path = 'E:/pythonProject/future/data/datafile/raw_feature/sw_industry_all.csv'
        self.dictionary = {}
        self.dictionary_saved_path = ''
        self.dictionary_load_path = ''
        self.industry_level1 = 'industry_level1'
        self.industry_level2 = 'industry_level2'
        self.industry_level3 = 'industry_level3'
    def get_industry_dictionary(self):
        industry_all = pd.read_csv(self.industry_all_path, encoding='utf-8')
        industry_dic = {}
        industry_level1_dic = {}
        industry_level1_dic['max'] = 0
        industry_dic[self.industry_level1] = industry_level1_dic
        industry_level2_dic = {}
        industry_level2_dic['max'] = 0
        industry_dic[self.industry_level2] = industry_level2_dic
        industry_level3_dic = {}
        industry_level3_dic['max'] = 0
        industry_dic[self.industry_level3] = industry_level3_dic
        for tup in industry_all.itertuples():
            industry_level1_dic = industry_dic[self.industry_level1]
            industry_level2_dic = industry_dic[self.industry_level2]
            industry_level3_dic = industry_dic[self.industry_level3]
            if not industry_level1_dic.__contains__(tup[2]):
                industry_level1_dic[tup[2]] = industry_level1_dic['max'] + 1
                industry_level1_dic['max'] = industry_level1_dic['max'] + 1
                industry_dic[self.industry_level1] = industry_level1_dic
            if not industry_level2_dic.__contains__(tup[3]):
                industry_level2_dic[tup[3]] = industry_level2_dic['max'] + 1
                industry_level2_dic['max'] = industry_level2_dic['max'] + 1
                industry_dic[self.industry_level2] = industry_level2_dic
            if not industry_level3_dic.__contains__(tup[4]):
                industry_level3_dic[tup[4]] = industry_level3_dic['max'] + 1
                industry_level3_dic['max'] = industry_level3_dic['max'] + 1
                industry_dic[self.industry_level3] = industry_level3_dic
        return industry_dic

    def dictionary_save(self):
        np.save(self.dictionary_saved_path, self.dictionary)
        return None

    def dictionary_load(self):
        dictionary = np.load(self.dictionary_load_path, allow_pickle=True).item()
        return dictionary

    def sw_code_to_bs_code(self, code):
        if code.startswith('00'):
            return 'sz.' + code
        elif code.startswith('200'):
            return 'sz.' + code
        elif code.startswith('30'):
            return 'sz.' + code
        elif code.startswith('60'):
            return 'sh.' + code
        elif code.startswith('68'):
            return 'sh.' + code
        elif code.startswith('8'):
            return 'bj.' + code
        elif code.startswith('430'):
            return 'bj.' + code
        else:
            return 'unknow.' + code

    def code_market(self, code):
        if code.startswith('sz.300'):
            return 3
        elif code.startswith('sh.60'):
            return 1
        elif code.startswith('sz.000'):
            return 2
        elif code.startswith('sh.688'):
            return 1
        elif code.startswith('sh.689'):
            return 1
        else:
            return 0

    def zh_code_market(self, code):
        if code.startswith('sz.399006'):
            return 3
        elif code.startswith('sh.000001'):
            return 1
        elif code.startswith('sz.399001'):
            return 2
        else:
            return 0

    def sw_date_to_bs_date(self, date):
        date_arr = date.split('/')
        year = date_arr[0]
        month = date_arr[1]
        day = date_arr[2]
        if len(month) == 1:
            month = '0' + month
        if len(day) == 1:
           day = '0' + day
        return year + '-' + month + '-' + day
    def hash_bucket(self, string_input, bucket_size):
        hash_code = hashlib.sha1(string_input.encode('utf-8')).hexdigest()
        int_num = int(hash_code, 16)
        index = int_num % bucket_size
        return index
    def get_recent_month_date(self, date, months):
        date_list = date.split('-')
        year = date_list[0]
        month = date_list[1]
        day = date_list[2]

        dt = datetime.date(int(year), int(month), int(day))
        month = dt.month - 1 + months
        year = dt.year + month // 12
        month = month % 12 + 1
        day = 1
        return str(dt.replace(year=year, month=month, day=day))

if __name__ == "__main__":
    tools = Tools()
    # industry_dic = tools.get_industry_dictionary()
    # tools.dictionary_saved_path = 'E:/pythonProject/future/dictionary/sw_industry_dic.npy'
    # tools.dictionary = industry_dic
    # tools.dictionary_save()
    # tools.dictionary_load_path = 'E:/pythonProject/future/dictionary/sw_industry_dic.npy'
    # dictionary = tools.dictionary_load()
    # print(dictionary)
    # print(tools.get_recent_month_date('2022-12-12', -9))