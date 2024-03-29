import csv
import gc

import baofuture as bs
import pandas as pd
import os
import copy
from feature.feature_process import *
#-*- coding: utf-8 -*-

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
                 date_start='1990-01-01',
                 date_end='2022-03-11'):
        self._bs = bs
        bs.login()
        self.date_start = date_start
        # self.date_end = datetime.datetime.now().strftime("%Y-%m-%d")
        self.date_end = date_end
        self.output_dir = output_dir
        self.fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        future_rs = bs.query_all_future(date)
        future_df = future_rs.get_data()
        print(future_df)
        return future_df
    def get_k_raw_data(self):
        future_df = self.get_codes_by_date(self.date_end)
        for index, row in future_df.iterrows():
            index += 1
            print(row["code"])
            print(index)
            code_k_data = bs.query_history_k_data_plus(row["code"], self.fields, start_date=self.date_start, end_date=self.date_end, frequency="d", adjustflag="1").get_data()
            if len(code_k_data)>0:
                if os.path.isfile('E:/pythonProject/future/data/datafile/code_k_data.csv'):
                    code_k_data.to_csv('{output_dir}/{code_k_data}.csv'.format(output_dir=self.output_dir, code_k_data='code_k_data'), mode='a', header=False, index=False)
                else:
                    code_k_data.to_csv('{output_dir}/{code_k_data}.csv'.format(output_dir=self.output_dir, code_k_data='code_k_data'),mode='a', header=True, index=False)

    def get_industry_data(self):
        rs = bs.query_future_industry()
        industry_list = []
        while (rs.error_code == '0') & rs.next():
            # print(rs.get_row_data())
            industry_list.append(rs.get_row_data())
        industry_list = pd.DataFrame(industry_list)
        industry_list.columns = ['date', 'code', 'code_name', 'industry', 'industry_level']
        industry_list = industry_list[['code', 'code_name', 'industry', 'industry_level']]
        industry_list_notnull = industry_list[industry_list["industry"] != '']
        industry_list.to_csv('{output_dir}/{industry_data}.csv'.format(output_dir=self.output_dir, industry_data='industry_data'), mode='a', header=True, index=False)

    def get_profit_data(self, code, year, quarter):
        # 查询季频估值指标盈利能力
        # ['code', 'pubDate', 'statDate', 'roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare']
        profit_list = []
        rs_profit = bs.query_profit_data(code=code, year=year, quarter=quarter)
        while (rs_profit.error_code == '0') & rs_profit.next():
            profit_list.append(rs_profit.get_row_data())
        result_profit = pd.DataFrame(profit_list, columns=rs_profit.fields)
        return result_profit

    def get_operation_data(self, code, year, quarter):
        # 季度营运能力
        operation_list = []
        rs_operation = bs.query_operation_data(code=code, year=year, quarter=quarter)
        while (rs_operation.error_code == '0') & rs_operation.next():
            operation_list.append(rs_operation.get_row_data())
        result_operation = pd.DataFrame(operation_list, columns=rs_operation.fields)
        return result_operation
    def get_growth_data(self, code, year, quarter):
        # 成长能力
        growth_list = []
        rs_growth = bs.query_growth_data(code=code, year=year, quarter=quarter)
        while (rs_growth.error_code == '0') & rs_growth.next():
            growth_list.append(rs_growth.get_row_data())
        result_growth = pd.DataFrame(growth_list, columns=rs_growth.fields)
        return result_growth

    def get_balance_data(self, code, year, quarter):
        # 偿债能力
        balance_list = []
        rs_balance = bs.query_balance_data(code=code, year=year, quarter=quarter)
        while (rs_balance.error_code == '0') & rs_balance.next():
            balance_list.append(rs_balance.get_row_data())
        result_balance = pd.DataFrame(balance_list, columns=rs_balance.fields)
        return result_balance

    def get_cash_data(self, code, year, quarter):
        # 季频现金流量
        cash_flow_list = []
        rs_cash_flow = bs.query_cash_flow_data(code=code, year=year, quarter=quarter)
        while (rs_cash_flow.error_code == '0') & rs_cash_flow.next():
            cash_flow_list.append(rs_cash_flow.get_row_data())
        result_cash_flow = pd.DataFrame(cash_flow_list, columns=rs_cash_flow.fields)
        return result_cash_flow

    def get_dupont_data(self, code, year, quarter):
        # 查询杜邦指数
        dupont_list = []
        rs_dupont = bs.query_dupont_data(code=code, year=year, quarter=quarter)
        while (rs_dupont.error_code == '0') & rs_dupont.next():
            dupont_list.append(rs_dupont.get_row_data())
        result_dupont = pd.DataFrame(dupont_list, columns=rs_dupont.fields)  # 打印输出
        return result_dupont

    def get_quarter_data(self, code, year, quarter):
        profit_data = self.get_profit_data(code, year, quarter)
        if not profit_data.empty:
            operation_data = self.get_operation_data(code, year, quarter)
            growth_data = self.get_growth_data(code, year, quarter)
            balance_data = self.get_balance_data(code, year, quarter)
            cash_data = self.get_cash_data(code, year, quarter)
            dupont_data = self.get_dupont_data(code, year, quarter)
            quarter_data = copy.deepcopy(profit_data[['code', 'pubDate', 'statDate']])
            quarter_data['roeAvg'] = profit_data['roeAvg'].map(lambda x: None if x == '' else float2Bucket(float(x) + 1, 1000, 0, 2, 2000))
            quarter_data['npMargin'] = profit_data['npMargin'].map(lambda x: None if x == '' else float2Bucket(float(x) + 1, 1000, 0, 2, 2000))
            quarter_data['gpMargin'] = profit_data['gpMargin'].map(lambda x: None if x == '' else float2Bucket(float(x) + 1, 1000, 0, 2, 2000))
            quarter_data['netProfit'] = profit_data['netProfit'].map(lambda x: None if x == '' else bignumber2Bucket(float(x) * 0.000001, 1.03, 1600, True))
            quarter_data['epsTTM'] = profit_data['epsTTM'].map(lambda x: None if x == '' else float2Bucket(float(x) + 10, 100, 0, 30, 3000))
            quarter_data['MBRevenue'] = profit_data['MBRevenue'].map(lambda x: None if x == '' else bignumber2Bucket(float(x) * 0.000001, 1.03, 800, False))
            quarter_data['totalShare'] = profit_data['totalShare'].map(lambda x: None if x == '' else bignumber2Bucket(float(x) * 0.0001, 1.03, 800, False))
            quarter_data['liqaShare'] = profit_data['liqaShare'].map(lambda x: None if x == '' else bignumber2Bucket(float(x) * 0.0001, 1.03, 800, False))
            quarter_data_tmp = pd.merge(profit_data, operation_data, how="left", left_on=['code', "pubDate"], right_on=['code', 'pubDate'])
            quarter_data['NRTurnRatio'] = quarter_data_tmp['NRTurnRatio'].map(lambda x: None if x == '' else float2Bucket(float(x), 10, 0, 100, 1000))
            # quarter_data['NRTurnDays'] = quarter_data_tmp['NRTurnDays']
            quarter_data['INVTurnRatio'] = quarter_data_tmp['INVTurnRatio'].map(lambda x: None if x == '' else float2Bucket(float(x), 10, 0, 100, 1000))
            # quarter_data['INVTurnDays'] = quarter_data_tmp['INVTurnDays']
            quarter_data['CATurnRatio'] = quarter_data_tmp['CATurnRatio'].map(lambda x: None if x == '' else float2Bucket(float(x), 20, 0, 5, 100))
            quarter_data['AssetTurnRatio'] = quarter_data_tmp['AssetTurnRatio'].map(lambda x: None if x == '' else float2Bucket(float(x), 20, 0, 5, 100))

            quarter_data_tmp = pd.merge(profit_data, growth_data, how="left", left_on=['code', "pubDate"],right_on=['code', 'pubDate'])
            quarter_data['YOYEquity'] = quarter_data_tmp['YOYEquity'].map(lambda x: None if x == '' else float2Bucket(float(x) + 1, 100, 0, 2, 200))
            quarter_data['YOYAsset'] = quarter_data_tmp['YOYAsset'].map(lambda x: None if x == '' else float2Bucket(float(x) + 1, 100, 0, 3, 300))
            quarter_data['YOYNI'] = quarter_data_tmp['YOYNI'].map(lambda x: None if x == '' else float2Bucket(float(x) + 10, 100, 0, 20, 2000))
            quarter_data['YOYEPSBasic'] = quarter_data_tmp['YOYEPSBasic'].map(lambda x: None if x == '' else float2Bucket(float(x) + 10, 100, 0, 20, 2000))
            quarter_data['YOYPNI'] = quarter_data_tmp['YOYPNI'].map(lambda x: None if x == '' else float2Bucket(float(x) + 10, 100, 0, 20, 2000))

            quarter_data_tmp = pd.merge(profit_data, balance_data, how="left", left_on=['code', "pubDate"],right_on=['code', 'pubDate'])
            quarter_data['currentRatio'] = quarter_data_tmp['currentRatio'].map(lambda x: None if x == '' else float2Bucket(float(x), 10, 0, 20, 200))
            quarter_data['quickRatio'] = quarter_data_tmp['quickRatio'].map(lambda x: None if x == '' else float2Bucket(float(x), 10, 0, 20, 200))
            quarter_data['cashRatio'] = quarter_data_tmp['cashRatio'].map(lambda x: None if x == '' else float2Bucket(float(x), 10, 0, 20, 200))
            quarter_data['YOYLiability'] = quarter_data_tmp['YOYLiability'].map(lambda x: None if x == '' else float2Bucket(float(x) + 5, 100, 0, 10, 1000))
            quarter_data['liabilityToAsset'] = quarter_data_tmp['liabilityToAsset'].map(lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 10, 1000))
            quarter_data['assetToEquity'] = quarter_data_tmp['assetToEquity'].map(lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 20, 2000))

            quarter_data_tmp = pd.merge(profit_data, cash_data, how="left", left_on=['code', "pubDate"],right_on=['code', 'pubDate'])
            quarter_data['CAToAsset'] = quarter_data_tmp['CAToAsset'].map(lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 1, 100))
            quarter_data['tangibleAssetToAsset'] = quarter_data_tmp['tangibleAssetToAsset'].map(lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 1, 100))
            quarter_data['ebitToInterest'] = quarter_data_tmp['ebitToInterest'].map(lambda x: None if x == '' else float2Bucket(float(x), 10, 0, 20, 200))
            quarter_data['CFOToOR'] = quarter_data_tmp['CFOToOR'].map(lambda x: None if x == '' else float2Bucket(float(x) + 5, 100, 0, 10, 1000))
            quarter_data['CFOToNP'] = quarter_data_tmp['CFOToNP'].map(lambda x: None if x == '' else float2Bucket(float(x) + 5, 100, 0, 10, 1000))
            quarter_data['CFOToGr'] = quarter_data_tmp['CFOToGr'].map(lambda x: None if x == '' else float2Bucket(float(x) + 5, 100, 0, 10, 1000))

            quarter_data_tmp = pd.merge(profit_data, dupont_data, how="left", left_on=['code', "pubDate"],right_on=['code', 'pubDate'])
            quarter_data['dupontROE'] = quarter_data_tmp['dupontROE'].map(lambda x: None if x == '' else float2Bucket(float(x) + 2, 100, 0, 4, 400))
            quarter_data['dupontAssetStoEquity'] = quarter_data_tmp['dupontAssetStoEquity'].map(lambda x: None if x == '' else float2Bucket(float(x), 100, 1, 11, 1000))
            quarter_data['dupontAssetTurn'] = quarter_data_tmp['dupontAssetTurn'].map(lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 1, 100))
            quarter_data['dupontPnitoni'] = quarter_data_tmp['dupontPnitoni'].map(lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 1, 100))
            quarter_data['dupontNitogr'] = quarter_data_tmp['dupontNitogr'].map(lambda x: None if x == '' else float2Bucket(float(x) + 1, 100, 0, 2, 200))
            quarter_data['dupontTaxBurden'] = quarter_data_tmp['dupontTaxBurden'].map(lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 1, 100))
            quarter_data['dupontIntburden'] = quarter_data_tmp['dupontIntburden'].map(lambda x: None if x == '' else float2Bucket(float(x), 100, 0, 1, 100))
            quarter_data['dupontEbittogr'] = quarter_data_tmp['dupontEbittogr'].map(lambda x: None if x == '' else float2Bucket(float(x) + 1, 100, 0, 2, 200))
            quarter_data['quarter'] = profit_data['pubDate'].map(lambda x: str(year) + str(quarter))
            quarter_data['pub_quarter'] = profit_data['pubDate'].map(lambda x: get_date_quarter(x.split('-')[0], x.split('-')[1], False))
            quarter_data['last_pub_quarter'] = profit_data['pubDate'].map(lambda x: get_date_quarter(x.split('-')[0], x.split('-')[1], True))
            return quarter_data
        else:
            return None

    def get_quarter_data_all(self, year_start, year_end):
        future_df = self.get_codes_by_date(self.date_end)
        # quarter_data_list = []
        index = 0
        for index, row in future_df.iterrows():
            quarter_data_list = []
            code = row["code"]
            index += 1
            if index >= 0:
                for year in range(year_start, year_end):
                    for quarter in range(4):
                        quarter_data = self.get_quarter_data(code, year, quarter + 1)
                        if quarter_data is not None:
                            quarter_data_list.append(quarter_data)
                if not quarter_data_list == []:
                    quarter_data_tmp = pd.concat(quarter_data_list)
                    if os.path.isfile('{output_dir}/{quarter_data}.csv'.format(output_dir=self.output_dir, quarter_data='quarter_data')):
                        quarter_data_tmp.to_csv('{output_dir}/{quarter_data}.csv'.format(output_dir=self.output_dir, quarter_data='quarter_data'), mode='a', header=False, index=False)
                    else:
                        quarter_data_tmp.to_csv('{output_dir}/{quarter_data}.csv'.format(output_dir=self.output_dir, quarter_data='quarter_data'), mode='a', header=True, index=False)


    def get_feature_and_label(self, k_file_name, industry_file_name, quarter_file_name):
        df = pd.read_csv(k_file_name)
        df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors='coerce')
        df = df[df['tradestatus'] == 1]
        df["open"] = pd.to_numeric(df["open"], errors='coerce')
        df["close"] = pd.to_numeric(df["close"], errors='coerce')
        df["pctChg"] = pd.to_numeric(df["pctChg"], errors='coerce')
        df["preclose"] = pd.to_numeric(df["preclose"], errors='coerce')
        df["high"] = pd.to_numeric(df["high"], errors='coerce')
        df["low"] = pd.to_numeric(df["low"], errors='coerce')
        df["turn"] = pd.to_numeric(df["turn"], errors='coerce')
        # df["open"] = pd.to_numeric(df["open"], errors='coerce')
        # df["open"] = pd.to_numeric(df["open"], errors='coerce')
        df["rank"] = df.groupby('code')['date'].rank(method="first")
        df['rank_1d'] = df['rank'].map(lambda x: x - 1)
        df['rank_3d'] = df['rank'].map(lambda x: x - 3)
        df['rank_5d'] = df['rank'].map(lambda x: x - 5)
        df['rank_7d'] = df['rank'].map(lambda x: x - 7)
        k_log_data = copy.deepcopy(df[["date", "code", "close", 'preclose', "rank", 'rank_1d', 'rank_3d', 'rank_5d', 'rank_7d']])
        label = copy.deepcopy(k_log_data[["date", "code"]])
        label["label"] = (k_log_data["close"] - k_log_data["preclose"])/k_log_data["preclose"] * 100
        for i in range(4):
            index = i * 2 + 1
            label_tmp = pd.merge(k_log_data, k_log_data, how="left", left_on=['code', "rank"], right_on=['code', "rank_" + str(index) + 'd'])
            label['label_' + str(index) + 'd'] = (label_tmp['close_y'] - label_tmp['preclose_x'])/label_tmp["preclose_x"]*100

        label.to_csv('{output_dir}/{label}.csv'.format(output_dir=self.output_dir, label='label'), mode='a', header=True, index=False)
        del label_tmp
        del label
        gc.collect()
        feature_date = copy.deepcopy(df[["date", "code", 'rank']])
        for i in range(4):
            feature_date['rank_' + str(i + 1) + 'd_fea'] = feature_date['rank'].map(lambda x: x + i + 1)
        for i in range(20):
            feature_date['rank_' + str(i * 3 + 7) + 'd_fea'] = feature_date['rank'].map(lambda x: x + i * 3 + 7)

        feature_date['open_ratio'] = ((df['open'] - df['preclose'])/df['preclose']).map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))
        feature_date['high_ratio'] = ((df['high'] - df['preclose']) / df['preclose']).map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))
        feature_date['low_ratio'] = ((df['low'] - df['preclose']) / df['preclose']).map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))
        feature_date['turn'] = df['turn'].map(lambda x: float2Bucket(float(x), 2000, 0, 1, 2000))
        feature_date['preclose'] = df['preclose'].map(lambda x: float2Bucket(float(x), 1, 0, 1000, 1000))
        feature_date['amount'] = df['amount'].map(lambda x: None if x == '' else float2Bucket(float(x) * 0.00000005, 1, 0, 40000, 40000))
        feature_date['pctChg'] = df['pctChg'].map(lambda x: float2Bucket(float(x) + 0.2, 1000, 0, 0.4, 400))
        feature_date['peTTM'] = df['peTTM'].map(lambda x: float2Bucket(float(x), 2, 0, 2000, 4000))
        feature_date['pcfNcfTTM'] = df['pcfNcfTTM'].map(lambda x: float2Bucket(float(x), 10, 0, 100, 1000))
        feature_date['pbMRQ'] = df['pbMRQ'].map(lambda x: float2Bucket(float(x), 10, 0, 500, 5000))
        feature_date['isST'] = df['isST'].map(lambda x: float2Bucket(float(x), 1, 0, 3, 3))
        feature_all = copy.deepcopy(feature_date[["date", "code", 'rank', 'open_ratio', 'high_ratio', 'low_ratio', 'turn', 'preclose', 'amount', 'pctChg', 'peTTM', 'pcfNcfTTM', 'pbMRQ', 'isST']])
        code_date = copy.deepcopy(feature_all[["date", "code"]])
        del df
        del k_log_data
        gc.collect()

        for i in range(4):
            fea_tmp = pd.merge(feature_date, feature_date, how="left", left_on=['code', "rank"], right_on=['code', 'rank_' + str(i + 1) + 'd_fea'])
            feature_all['open_ratio_' + str(i + 1) + 'd'] = fea_tmp['open_ratio_y']
            feature_all['high_ratio_' + str(i + 1) + 'd'] = fea_tmp['high_ratio_y']
            feature_all['low_ratio_' + str(i + 1) + 'd'] = fea_tmp['low_ratio_y']
            feature_all['turn_' + str(i + 1) + 'd'] = fea_tmp['turn_y']
            feature_all['pctChg_' + str(i + 1) + 'd'] = fea_tmp['pctChg_y']
        for i in range(20):
            fea_tmp = pd.merge(feature_date, feature_date, how="left", left_on=['code', "rank"], right_on=['code', 'rank_' + str(i * 3 + 7) + 'd_fea'])
            feature_all['open_ratio_' + str(i * 3 + 7) + 'd'] = fea_tmp['open_ratio_y']
            feature_all['high_ratio_' + str(i * 3 + 7) + 'd'] = fea_tmp['high_ratio_y']
            feature_all['low_ratio_' + str(i * 3 + 7) + 'd'] = fea_tmp['low_ratio_y']
            feature_all['turn_' + str(i * 3 + 7) + 'd'] = fea_tmp['turn_y']
            feature_all['pctChg_' + str(i * 3 + 7) + 'd'] = fea_tmp['pctChg_y']
        del feature_date
        del fea_tmp
        gc.collect()
        # 行业特征
        industry_data = pd.read_csv(industry_file_name)
        fea_tmp = pd.merge(code_date, industry_data, how="left", left_on=['code'],right_on=['code'])
        feature_all['industry'] = fea_tmp['industry']
        del industry_data
        del fea_tmp
        gc.collect()

        # 季度财报特征
        quarter_data_all = pd.read_csv(quarter_file_name)
        quarter_data_final = pd.merge(quarter_data_all, quarter_data_all, how="left", left_on=['code', "pub_quarter"], right_on=['code', 'last_pub_quarter'])
        code_date['quarter'] = pd.to_numeric(code_date['date'].map(lambda x: get_date_quarter(x.split('-')[0], x.split('-')[1], False)), errors='coerce')
        quarter_data_final["pub_quarter_x"] = pd.to_numeric(quarter_data_final["pub_quarter_x"], errors='coerce')
        fea_tmp = pd.merge(code_date, quarter_data_final, how="left", left_on=['code', 'quarter'], right_on=['code', 'pub_quarter_x'])
        fea_list = ['roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare', 'NRTurnRatio', 'INVTurnRatio', 'CATurnRatio', 'AssetTurnRatio', 'YOYEquity', 'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI', 'currentRatio', 'quickRatio', 'cashRatio', 'YOYLiability', 'liabilityToAsset', 'assetToEquity', 'CAToAsset', 'tangibleAssetToAsset', 'ebitToInterest', 'CFOToOR', 'CFOToNP', 'CFOToGr', 'dupontROE', 'dupontAssetStoEquity', 'dupontAssetTurn', 'dupontPnitoni', 'dupontNitogr', 'dupontTaxBurden', 'dupontIntburden', 'dupontEbittogr']
        del quarter_data_all
        del quarter_data_final
        gc.collect()
        for fea in fea_list:
            print(fea)
            feature_all[fea] = fea_tmp[['date', 'pubDate_x', fea + '_x', fea + '_y']].apply(lambda x: x[fea + '_x'] if str(x.date) > str(x.pubDate_x) else x[fea + '_y'], axis=1)
        feature_all.to_csv('{output_dir}/{feature_all}.csv'.format(output_dir=self.output_dir, feature_all='feature_all'), mode='a', header=True, index=False)

    def get_sample(self, feature_file, label_file):
        # label = pd.read_csv(label_file)
        # normal_label = copy.deepcopy(label[['date', 'code']])
        # normal_label['label'] = label['label'].map(lambda x: get_classification_label(x))
        # normal_label['label_1d'] = label['label_1d'].map(lambda x: get_classification_label(x))
        # normal_label['label_3d'] = label['label_3d'].map(lambda x: get_classification_label(x))
        # normal_label['label_5d'] = label['label_5d'].map(lambda x: get_classification_label(x))
        # normal_label['label_7d'] = label['label_7d'].map(lambda x: get_classification_label(x))
        # normal_label = normal_label[normal_label['label_7d'] >= 0]
        # del label
        # gc.collect()
        # feature = pd.read_csv(feature_file)
        # all_data = pd.merge(feature, normal_label, how="right", left_on=['code', "date"],right_on=['code', 'date'])
        # del feature
        # gc.collect()
        # train_data = all_data[all_data['date']<'2021-01-01']
        # test_data = all_data[all_data['date']>='2021-01-01']
        # del all_data
        # gc.collect()
        # train_data = train_data.sort_values(by=['date', 'code'], ascending=True)
        # test_data = test_data.sort_values(by=['date', 'code'], ascending=True)
        # train_data = pd.read_csv('E:/pythonProject/future/data/datafile/train_data_category.csv')
        # train_data.to_csv('{output_dir}/{train_data}.csv'.format(output_dir=self.output_dir, train_data='train_data_category1'), mode='a',header=True, index=False, encoding='utf-8')
        # del train_data
        # gc.collect()
        # f = open('{output_dir}/{test_data}.csv'.format(output_dir=self.output_dir, test_data='test_data_category1'), mode='a', encoding='utf-8', newline="")
        # for i in range(5000):
        #     if i > 4990:
        #         print(f.readline())
        #         # with open('{output_dir}/{test_data}.csv'.format(output_dir=self.output_dir, test_data='test_data_category1'), mode='a', encoding='utf-8', newline="") as f:
        #     else:
        #         f.readline()
        # f = open('{output_dir}/{test_data}.csv'.format(output_dir=self.output_dir, test_data='test_data_category1'), mode='a', encoding='utf-8', newline="")
        # test_data = pd.read_csv('E:/pythonProject/future/data/datafile/test_data_category1.csv')
        # test_data = test_data.iloc[0:10000,:]
        # test_data.to_csv('{output_dir}/{train_data}.csv'.format(output_dir=self.output_dir, train_data='test_data_category1'), mode='a',header=True, index=False, encoding='utf-8')

        with open('E:/pythonProject/future/data/datafile/egg.csv', mode='w+', encoding='utf-8-sig', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            spamwriter.writerow(['a', '1', '1', '2', '2'])
            spamwriter.writerow(['b', '3', '3', '6', '4'])
            spamwriter.writerow(['c', '7', '7', '10', '4'])
            spamwriter.writerow(['d', '11', '11', '11', '1'])
            spamwriter.writerow(['e', '12', '12', '14', '3'])



if __name__ == '__main__':
    mkdir('./datafile')
    downloader = Downloader('./datafile/', date_start='2019-01-01', date_end='2022-04-21')
    k_file_name = 'E:/pythonProject/future/data/datafile/code_k_data.csv'
    industry_file_name = 'E:/pythonProject/future/data/datafile/industry_data.csv'
    quarter_file_name = 'E:/pythonProject/future/data/datafile/quarter_data.csv'
    feature_file = 'E:/pythonProject/future/data/datafile/feature_all.csv'
    label_file = 'E:/pythonProject/future/data/datafile/label.csv'
    # downloader.get_k_raw_data()
    # downloader.get_industry_data()
    # downloader.get_quarter_data_all(2018, 2022)
    # downloader.get_feature_and_label(k_file_name, industry_file_name, quarter_file_name)
    downloader.get_sample(feature_file, label_file)