import csv
import gc
import multiprocessing

import baostock as bs
import pandas as pd
import os
import copy

from feature.feature_process import *
import logging
from logging import handlers
from tools.path_enum import Path

#-*- coding: utf-8 -*-

import logging
from logging import handlers
import pandas as pd
import numpy as np
industry_dict = np.load('E:/pythonProject/stock/common/industry_dict.npy').item()

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,printflag=False,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        if printflag:
            sh = logging.StreamHandler()
            sh.setFormatter(format_str)
            self.logger.addHandler(sh)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(format_str)
        self.logger.addHandler(th)

# 省略问题修改配置, 打印100列数据
pd.set_option('display.max_columns', 200)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)

OUTPUT = './datafile'


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Downloader(object):
    def __init__(self,
                 output_dir,
                 date_start='2007-10-01',
                 date_end='2022-03-11'):
        self._bs = bs
        bs.login()
        self.date_start = date_start
        # self.date_end = datetime.datetime.now().strftime("%Y-%m-%d")
        self.date_end = date_end
        self.output_dir = output_dir
        self.data_version = 'v5'
        self.his_data_version = 'v4'
        self.code_quarter_data_file_name = 'code_quarter_data_v2_2007'
        self.fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
        self.stock_df = self.get_codes_by_date(date_end)
        self.sw_code_all_industry_path = Path.sw_code_all_industry_path
        self.his_raw_data_path = Path.raw_feature + self.his_data_version + '_' + self.date_start.split('-')[0] + '.csv'
        self.raw_data_save_path = Path.raw_feature + self.data_version + '_' + self.date_start.split('-')[0] + '.csv'
        self.code_industry_all = pd.read_csv(self.sw_code_all_industry_path, encoding='utf-8')
        self.code_industry = self.code_industry_all
        self.code_industry['start_date'] = pd.to_datetime(self.code_industry['start_date'])
        self.code_industry['row_num'] = self.code_industry.groupby(['code'])['start_date'].rank(ascending=False, method='first').astype(int)
        self.code_industry = self.code_industry[self.code_industry['row_num'] == 1]
        self.code_industry = self.code_industry[['code','industry_name_level1','industry_name_level2','industry_name_level3','industry_id_level1','industry_id_level2','industry_id_level3']]
    def exit(self):
        bs.logout()

    # 获取当天的所有可交易的股票dataframe，包含code，tradeStatus，code_name
    def get_codes_by_date(self, date):
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        return stock_df

    def get_k_raw_data(self):
        # stock_df是指当天的所有可交易的股票dataframe
        df_list = []
        for index, row in self.stock_df.iterrows():
            index += 1
            code_k_data = bs.query_history_k_data_plus(row["code"], self.fields, start_date=self.date_start, end_date=self.date_end, frequency="d", adjustflag="2").get_data()
            if len(code_k_data)>0:
                code_k_data = pd.merge(code_k_data, self.code_industry,  how="left", left_on=['code'], right_on=['code'])
                code_k_data['industry_name_level1'] = code_k_data['industry_name_level1'].map(lambda x: x if type(x) == str else 'level1_default')
                code_k_data['industry_name_level2'] = code_k_data['industry_name_level2'].map(lambda x: x if type(x) == str else 'level2_default')
                code_k_data['industry_name_level3'] = code_k_data['industry_name_level3'].map(lambda x: x if type(x) == str else 'level3_default')
                code_k_data['industry_id_level1'] = code_k_data['industry_id_level1'].map(lambda x: x if x>0 else 0)
                code_k_data['industry_id_level2'] = code_k_data['industry_id_level2'].map(lambda x: x if x>0 else 0)
                code_k_data['industry_id_level3'] = code_k_data['industry_id_level3'].map(lambda x: x if x>0 else 0)
                if os.path.isfile(self.raw_data_save_path):
                    code_k_data.to_csv(self.raw_data_save_path, mode='a', header=False, index=False)
                else:
                    code_k_data.to_csv(self.raw_data_save_path,mode='a', header=True, index=False)

    def get_his_raw_data(self):
        code_industry = self.code_industry_all
        his_data = pd.read_csv(self.his_raw_data_path, encoding='utf-8')
        his_data_sw = pd.merge(his_data, code_industry, how="left", left_on=['code'], right_on=['code'])
        his_data_sw_tmp = his_data_sw[his_data_sw['start_date'].map(lambda x: type(x) != str)]

        his_data_sw1 = his_data_sw[his_data_sw['date']>his_data_sw['start_date']]
        his_data_sw1['start_date'] = pd.to_datetime(his_data_sw1['start_date'])
        his_data_sw1['row_num'] = his_data_sw1.groupby(['date', 'code'])['start_date'].rank(ascending=False,method='first').astype(int)
        his_data_sw1 = his_data_sw1[his_data_sw1['row_num'] == 1]

        his_data_sw2 = his_data_sw[his_data_sw['date'] <= his_data_sw['start_date']]
        his_data_sw2['start_date'] = pd.to_datetime(his_data_sw2['start_date'])
        his_data_sw2['row_num'] = his_data_sw2.groupby(['date', 'code'])['start_date'].rank(ascending=True, method='first').astype(int)
        his_data_sw2 = his_data_sw2[his_data_sw2['row_num'] == 1]

        his_data_sw = pd.concat([his_data_sw1, his_data_sw2], axis=0)
        his_data_sw['row_num'] = his_data_sw.groupby(['date', 'code'])['start_date'].rank(ascending=True, method='first').astype(int)
        his_data_sw = his_data_sw[his_data_sw['row_num'] == 1]

        his_data_sw = pd.concat([his_data_sw, his_data_sw_tmp], axis=0)
        his_data_sw = his_data_sw[['date','code','open','high','low','close','preclose','volume','amount','adjustflag','turn','tradestatus','pctChg','peTTM','pbMRQ','psTTM','pcfNcfTTM','isST','industry_name_level1','industry_name_level2','industry_name_level3','industry_id_level1','industry_id_level2','industry_id_level3']]
        his_data_sw = his_data_sw.sort_values(by = ['date','code'])
        his_data_sw['industry_name_level1'] = his_data_sw['industry_name_level1'].map(lambda x: x if type(x) == str else 'level1_default')
        his_data_sw['industry_name_level2'] = his_data_sw['industry_name_level2'].map(lambda x: x if type(x) == str else 'level2_default')
        his_data_sw['industry_name_level3'] = his_data_sw['industry_name_level3'].map(lambda x: x if type(x) == str else 'level3_default')
        his_data_sw['industry_id_level1'] = his_data_sw['industry_id_level1'].map(lambda x: x if x>0 else 0)
        his_data_sw['industry_id_level2'] = his_data_sw['industry_id_level2'].map(lambda x: x if x>0 else 0)
        his_data_sw['industry_id_level3'] = his_data_sw['industry_id_level3'].map(lambda x: x if x>0 else 0)
        return his_data_sw
    def save_raw_data(self, raw_data, save_path, header=True):
        raw_data.to_csv(save_path, mode='a', header=header, index=False)

def job(q):
    import datetime
    arg =  q.get()
    year = int(arg.split('-')[0])
    month = int(arg.split('-')[1])
    day = int(arg.split('-')[2])
    # year = q.get()
    begin = datetime.date(year, month, day)
    end = datetime.date(year, 12, 31)
    d = begin
    delta = datetime.timedelta(days=1)
    while d <= end:
        logpath = 'E:/pythonProject/future/data/datafile/raw_feature/' + str(year) + '_log.txt'
        log = Logger(logpath, level='info')
        log.logger.info(d.strftime("%Y-%m-%d" + ":  start !!!!!!!!!!"))
        downloader = Downloader('/data/datafile/raw_feature', date_start=d.strftime("%Y-%m-%d"),
                                date_end=d.strftime("%Y-%m-%d"))
        if len(downloader.stock_df) > 0:
            downloader.get_k_raw_data()

        downloader.exit()
        log.logger.info(d.strftime("%Y-%m-%d" + ":  DONE !!!!!!!!!!"))
        d += delta

def main():
    q = multiprocessing.JoinableQueue()
    pw1 = multiprocessing.Process(target=job, args=(q,))
    # pw2 = multiprocessing.Process(target=job, args=(q,))
    pw1.daemon = True
    # pw2.daemon = True
    pw1.start()
    # pw2.start()
    for year in ['2022-11-18']:
        q.put(year)
    try:
        q.join()
    except KeyboardInterrupt:
        print("stop by hands")
if __name__ == '__main__':
    main()
    # dates = ['2007-10-01','2008-10-01','2009-10-01','2010-10-01','2011-10-01','2012-10-01','2013-10-01','2014-10-01','2015-10-01','2016-10-01','2017-10-01','2018-10-01','2019-10-01','2020-10-01','2021-10-01','2022-10-01']
    # dates = ['2022-10-01']
    # for date in dates:
    #     downloader = Downloader('/data/datafile/raw_feature', date_start=date)
        # his_data_sw = downloader.get_his_raw_data()
        # downloader.save_raw_data(his_data_sw, downloader.raw_data_save_path, True)
        # code_industry = pd.read_csv(downloader.his_raw_data_path, encoding='utf-8')
        # code_industry1 = pd.read_csv(downloader.raw_data_save_path, encoding='utf-8')
        # print(code_industry.shape)
        # print(code_industry1.shape)

