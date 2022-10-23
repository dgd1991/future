import baofuture as bs
import pandas as pd
import os
import copy
from feature.feature_process import *

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
        df_list = []
        index = 0
        for index, row in future_df.iterrows():
            index += 1
            print(row["code"])
            print(index)
            code_k_data = bs.query_history_k_data_plus(row["code"], self.fields, start_date=self.date_start, end_date=self.date_end, frequency="d", adjustflag="1").get_data()
            if len(code_k_data)>0:
                if os.path.isfile('E:/pythonProject/future/datafile/code_k_data.csv'):
                    code_k_data.to_csv('{output_dir}/{code_k_data}.csv'.format(output_dir=self.output_dir, code_k_data='code_k_data'), mode='a', header=False, index=False)
                else:
                    code_k_data.to_csv('{output_dir}/{code_k_data}.csv'.format(output_dir=self.output_dir, code_k_data='code_k_data'),mode='a', header=True, index=False)

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
        label = k_log_data[["date", "code"]]
        label["label"] = (k_log_data["close"] - k_log_data["preclose"])/k_log_data["preclose"]*100
        for i in range(4):
            index = i * 2 + 1
            label_tmp = pd.merge(k_log_data, k_log_data, how="left", left_on=['code', "rank"], right_on=['code', "rank_" + str(index) + 'd'])
            label['label_' + str(index) + 'd'] = (label_tmp['close_y'] - label_tmp['preclose_x'])/label_tmp["preclose_x"]*100

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
        # print(feature_all.iloc[800:802:])
        # print(label.iloc[800:802])
        # 行业特征
        industry_data = pd.read_csv(industry_file_name)
        fea_tmp = pd.merge(code_date, industry_data, how="left", left_on=['code'],right_on=['code'])
        feature_all['industry'] = fea_tmp['industry']

        # 季度财报特征
        quarter_data_all = pd.read_csv(quarter_file_name)
        column_name = 'code,pubDate,statDate,roeAvg,npMargin,gpMargin,netProfit,epsTTM,MBRevenue,totalShare,liqaShare,NRTurnRatio,INVTurnRatio,CATurnRatio,AssetTurnRatio,YOYEquity,YOYAsset,YOYNI,YOYEPSBasic,YOYPNI,currentRatio,quickRatio,cashRatio,YOYLiability,liabilityToAsset,assetToEquity,CAToAsset,tangibleAssetToAsset,ebitToInterest,CFOToOR,CFOToNP,CFOToGr,dupontROE,dupontAssetStoEquity,dupontAssetTurn,dupontPnitoni,dupontNitogr,dupontTaxBurden,dupontIntburden,dupontEbittogr,quarter,pub_quarter,last_pub_quarter'
        quarter_data_all.columns = column_name.split(',')
        quarter_data_final = pd.merge(quarter_data_all, quarter_data_all, how="left", left_on=['code', "pub_quarter"], right_on=['code', 'last_pub_quarter'])
        code_date['quarter'] = pd.to_numeric(code_date['date'].map(lambda x: get_date_quarter(x.split('-')[0], x.split('-')[1], False)), errors='coerce')
        quarter_data_final["pub_quarter_x"] = pd.to_numeric(quarter_data_final["pub_quarter_x"], errors='coerce')
        fea_tmp = pd.merge(code_date, quarter_data_final, how="left", left_on=['code', 'quarter'], right_on=['code', 'pub_quarter_x'])
        # ['roeAvg','npMargin','gpMargin','netProfit','epsTTM','MBRevenue','totalShare','liqaShare','NRTurnRatio','INVTurnRatio','CATurnRatio','AssetTurnRatio','YOYEquity','YOYAsset','YOYNI','YOYEPSBasic','YOYPNI','currentRatio','quickRatio','cashRatio','YOYLiability','liabilityToAsset','assetToEquity','CAToAsset','tangibleAssetToAsset','ebitToInterest','CFOToOR','CFOToNP','CFOToGr','dupontROE','dupontAssetStoEquity','dupontAssetTurn','dupontPnitoni','dupontNitogr','dupontTaxBurden','dupontIntburden','dupontEbittogr']
        feature_all['roeAvg'] = fea_tmp.apply(lambda x: x.roeAvg_x if str(x.date) > str(x.pubDate_x) else x.roeAvg_y, axis=1)
        feature_all['npMargin'] = fea_tmp.apply(lambda x: x.npMargin_x if str(x.date) > str(x.pubDate_x) else x.npMargin_y, axis=1)
        feature_all['gpMargin'] = fea_tmp.apply(lambda x: x.gpMargin_x if str(x.date) > str(x.pubDate_x) else x.gpMargin_y, axis=1)
        feature_all['netProfit'] = fea_tmp.apply(lambda x: x.netProfit_x if str(x.date) > str(x.pubDate_x) else x.netProfit_y, axis=1)
        feature_all['epsTTM'] = fea_tmp.apply(lambda x: x.epsTTM_x if str(x.date) > str(x.pubDate_x) else x.epsTTM_y, axis=1)
        feature_all['MBRevenue'] = fea_tmp.apply(lambda x: x.MBRevenue_x if str(x.date) > str(x.pubDate_x) else x.MBRevenue_y, axis=1)
        feature_all['totalShare'] = fea_tmp.apply(lambda x: x.totalShare_x if str(x.date) > str(x.pubDate_x) else x.totalShare_y, axis=1)
        feature_all['liqaShare'] = fea_tmp.apply(lambda x: x.liqaShare_x if str(x.date) > str(x.pubDate_x) else x.liqaShare_y, axis=1)
        feature_all['NRTurnRatio'] = fea_tmp.apply(lambda x: x.NRTurnRatio_x if str(x.date) > str(x.pubDate_x) else x.NRTurnRatio_y, axis=1)
        feature_all['INVTurnRatio'] = fea_tmp.apply(lambda x: x.INVTurnRatio_x if str(x.date) > str(x.pubDate_x) else x.INVTurnRatio_y, axis=1)
        feature_all['CATurnRatio'] = fea_tmp.apply(lambda x: x.CATurnRatio_x if str(x.date) > str(x.pubDate_x) else x.CATurnRatio_y, axis=1)
        feature_all['AssetTurnRatio'] = fea_tmp.apply(lambda x: x.AssetTurnRatio_x if str(x.date) > str(x.pubDate_x) else x.AssetTurnRatio_y, axis=1)
        feature_all['YOYEquity'] = fea_tmp.apply(lambda x: x.YOYEquity_x if str(x.date) > str(x.pubDate_x) else x.YOYEquity_y, axis=1)
        feature_all['YOYAsset'] = fea_tmp.apply(lambda x: x.YOYAsset_x if str(x.date) > str(x.pubDate_x) else x.YOYAsset_y, axis=1)
        feature_all['YOYNI'] = fea_tmp.apply(lambda x: x.YOYNI_x if str(x.date) > str(x.pubDate_x) else x.YOYNI_y, axis=1)
        feature_all['YOYEPSBasic'] = fea_tmp.apply(lambda x: x.YOYEPSBasic_x if str(x.date) > str(x.pubDate_x) else x.YOYEPSBasic_y, axis=1)
        feature_all['YOYPNI'] = fea_tmp.apply(lambda x: x.YOYPNI_x if str(x.date) > str(x.pubDate_x) else x.YOYPNI_y, axis=1)
        feature_all['currentRatio'] = fea_tmp.apply(lambda x: x.currentRatio_x if str(x.date) > str(x.pubDate_x) else x.currentRatio_y, axis=1)
        feature_all['quickRatio'] = fea_tmp.apply(lambda x: x.quickRatio_x if str(x.date) > str(x.pubDate_x) else x.quickRatio_y, axis=1)
        feature_all['cashRatio'] = fea_tmp.apply(lambda x: x.cashRatio_x if str(x.date) > str(x.pubDate_x) else x.cashRatio_y, axis=1)
        feature_all['YOYLiability'] = fea_tmp.apply(lambda x: x.YOYLiability_x if str(x.date) > str(x.pubDate_x) else x.YOYLiability_y, axis=1)
        feature_all['liabilityToAsset'] = fea_tmp.apply(lambda x: x.liabilityToAsset_x if str(x.date) > str(x.pubDate_x) else x.liabilityToAsset_y, axis=1)
        feature_all['assetToEquity'] = fea_tmp.apply(lambda x: x.assetToEquity_x if str(x.date) > str(x.pubDate_x) else x.assetToEquity_y, axis=1)
        feature_all['CAToAsset'] = fea_tmp.apply(lambda x: x.CAToAsset_x if str(x.date) > str(x.pubDate_x) else x.CAToAsset_y, axis=1)
        feature_all['tangibleAssetToAsset'] = fea_tmp.apply(lambda x: x.tangibleAssetToAsset_x if str(x.date) > str(x.pubDate_x) else x.tangibleAssetToAsset_y, axis=1)
        feature_all['ebitToInterest'] = fea_tmp.apply(lambda x: x.ebitToInterest_x if str(x.date) > str(x.pubDate_x) else x.ebitToInterest_y, axis=1)
        feature_all['CFOToOR'] = fea_tmp.apply(lambda x: x.CFOToOR_x if str(x.date) > str(x.pubDate_x) else x.CFOToOR_y, axis=1)
        feature_all['CFOToNP'] = fea_tmp.apply(lambda x: x.CFOToNP_x if str(x.date) > str(x.pubDate_x) else x.CFOToNP_y, axis=1)
        feature_all['CFOToGr'] = fea_tmp.apply(lambda x: x.CFOToGr_x if str(x.date) > str(x.pubDate_x) else x.CFOToGr_y, axis=1)
        feature_all['dupontROE'] = fea_tmp.apply(lambda x: x.dupontROE_x if str(x.date) > str(x.pubDate_x) else x.dupontROE_y, axis=1)
        feature_all['dupontAssetStoEquity'] = fea_tmp.apply(lambda x: x.dupontAssetStoEquity_x if str(x.date) > str(x.pubDate_x) else x.dupontAssetStoEquity_y, axis=1)
        feature_all['dupontAssetTurn'] = fea_tmp.apply(lambda x: x.dupontAssetTurn_x if str(x.date) > str(x.pubDate_x) else x.dupontAssetTurn_y, axis=1)
        feature_all['dupontPnitoni'] = fea_tmp.apply(lambda x: x.dupontPnitoni_x if str(x.date) > str(x.pubDate_x) else x.dupontPnitoni_y, axis=1)
        feature_all['dupontNitogr'] = fea_tmp.apply(lambda x: x.dupontNitogr_x if str(x.date) > str(x.pubDate_x) else x.dupontNitogr_y, axis=1)
        feature_all['dupontTaxBurden'] = fea_tmp.apply(lambda x: x.dupontTaxBurden_x if str(x.date) > str(x.pubDate_x) else x.dupontTaxBurden_y, axis=1)
        feature_all['dupontIntburden'] = fea_tmp.apply(lambda x: x.dupontIntburden_x if str(x.date) > str(x.pubDate_x) else x.dupontIntburden_y, axis=1)
        feature_all['dupontEbittogr'] = fea_tmp.apply(lambda x: x.dupontEbittogr_x if str(x.date) > str(x.pubDate_x) else x.dupontEbittogr_y, axis=1)
        feature_all.to_csv('{output_dir}/{feature_all}.csv'.format(output_dir=self.output_dir, feature_all='feature_all'), mode='a', header=True, index=False)
        label.to_csv('{output_dir}/{label}.csv'.format(output_dir=self.output_dir, label='label'), mode='a', header=True, index=False)
        # print(label.iloc[800:802])
        return feature_all, label

    # def read_history_data(self):
    #     raw_data = pd.read_csv("E:/pythonProject/future/datafile/sz.399998.中证煤炭指数.csv")
    #     raw_data['date_order'] = raw_data.groupby(['code']).date.rank(method='first', ascending=True)

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
            # print(operation_data)
            # print(growth_data)
            # print(balance_data)
            # print(cash_data)
            # print(dupont_data)

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
            if index > 5012:
                for year in range(year_start, year_end):
                    for quarter in range(4):
                        quarter_data = self.get_quarter_data(code, year, quarter + 1)
                        if quarter_data is not None:
                            quarter_data_list.append(quarter_data)
                if not quarter_data_list == []:
                    quarter_data_tmp = pd.concat(quarter_data_list)
                    if index == 1:
                        quarter_data_tmp.to_csv('{output_dir}/{quarter_data}.csv'.format(output_dir=self.output_dir, quarter_data='quarter_data'), mode='a', header=True, index=False)
                    else:
                        quarter_data_tmp.to_csv('{output_dir}/{quarter_data}.csv'.format(output_dir=self.output_dir, quarter_data='quarter_data'), mode='a', header=False, index=False)
                quarter_data_all = pd.concat(quarter_data_list)
        # print(quarter_data_all.columns)
        # quarter_data_final = pd.merge(quarter_data_all, quarter_data_all, how="left", left_on=['code', "pub_quarter"],right_on=['code', 'last_pub_quarter'])
        # # x: last_quarter, y：last_last_quarter
        # quarter_data_final.to_csv('{output_dir}/{quarter_data_final}.csv'.format(output_dir=self.output_dir, data='quarter_data_final'), index=False)
        # # print(quarter_data_final.iloc[0:3])
        # # print(quarter_data_final.count())
        # return quarter_data_final

    def get_sample(self, feature_file, label_file):
        feature = pd.read_csv(feature_file)
        label = pd.read_csv(label_file)
        normal_label = copy.deepcopy(label[['date', 'code']])
        # label, label_1d, label_3d, label_5d, label_7d
        # normal_label['label'] = label['label'].map(lambda x: get_normal_label(x))
        # normal_label['label_1d'] = label['label_1d'].map(lambda x: get_normal_label(x))
        # normal_label['label_3d'] = label['label_3d'].map(lambda x: get_normal_label(x))
        # normal_label['label_5d'] = label['label_5d'].map(lambda x: get_normal_label(x))
        # normal_label['label_7d'] = label['label_7d'].map(lambda x: get_normal_label(x))
        normal_label['label'] = label['label'].map(lambda x: get_classification_label(x))
        normal_label['label_1d'] = label['label_1d'].map(lambda x: get_classification_label(x))
        normal_label['label_3d'] = label['label_3d'].map(lambda x: get_classification_label(x))
        normal_label['label_5d'] = label['label_5d'].map(lambda x: get_classification_label(x))
        normal_label['label_7d'] = label['label_7d'].map(lambda x: get_classification_label(x))
        normal_label = normal_label[normal_label['label_7d'] >= 0]
        all_data = pd.merge(feature, normal_label, how="right", left_on=['code', "date"],right_on=['code', 'date'])
        train_data = all_data[all_data['date']<'2021-01-01']
        test_data = all_data[all_data['date']>='2021-01-01']
        train_data = train_data.sort_values(by=['date', 'code'], ascending=True)
        test_data = test_data.sort_values(by=['date', 'code'], ascending=True)
        train_data.to_csv('{output_dir}/{train_data}.csv'.format(output_dir=self.output_dir, train_data='train_data_category'), mode='a',header=True, index=False)
        test_data.to_csv('{output_dir}/{test_data}.csv'.format(output_dir=self.output_dir, test_data='test_data_category'), mode='a',header=True, index=False)


if __name__ == '__main__':
    mkdir('./datafile')
    downloader = Downloader('./datafile/', date_start='2019-01-01', date_end='2022-04-21')
    k_file_name = 'E:/pythonProject/future/datafile/code_k_data.csv'
    industry_file_name = 'E:/pythonProject/future/datafile/industry_data.csv'
    quarter_file_name = 'E:/pythonProject/future/datafile/quarter_data.csv'
    feature_file = 'E:/pythonProject/future/datafile/feature_all.csv'
    label_file = 'E:/pythonProject/future/datafile/label.csv'
    # downloader.get_k_data()
    # downloader.get_quarter_data(code="sh.600000", year=2017, quarter=2)
    # downloader.get_quarter_data_all(1991, 2022)
    downloader.get_industry_data()
    # downloader.get_k_raw_data()
    # downloader.get_feature_and_label(k_file_name, industry_file_name, quarter_file_name)
    # downloader.get_sample(feature_file, label_file)
    #
    # mkdir('./datafile/test')
    # downloader = Downloader('./datafile/test', date_start='2022-03-01', date_end='2022-03-11')
    # downloader.run()