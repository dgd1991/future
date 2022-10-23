import pandas as pd


pd.set_option('display.max_columns', 200)

# 截断问题修改配置，每行展示数据的宽度为230
pd.set_option('display.width', 230)

class Label(object):
	def __init__(self, path1, path2, year):
		self.path1 = path1
		self.path2 = path2
		self.year = year
	def get_raw_label(self):
		code_k_data_path1 = pd.read_csv(self.path1)
		code_k_data_path2 = pd.read_csv(self.path2)
		code_k_data_both = pd.concat([code_k_data_path1, code_k_data_path2], axis=0)
		# code_k_data_both = code_k_data_path1

		code_k_data_both["tradestatus"] = pd.to_numeric(code_k_data_both["tradestatus"], errors='coerce')
		code_k_data_both["turn"] = pd.to_numeric(code_k_data_both["turn"], errors='coerce')
		code_k_data_both["pctChg"] = pd.to_numeric(code_k_data_both["pctChg"], errors='coerce')
		code_k_data_both = code_k_data_both[(code_k_data_both['tradestatus'] == 1) & (code_k_data_both['turn'] > 0) & (code_k_data_both['pctChg'] <= 20) & (code_k_data_both['pctChg'] >= -20)]
		code_k_data_both = code_k_data_both[['date', 'code', 'close']]
		code_k_data_both["close"] = pd.to_numeric(code_k_data_both["close"], errors='coerce')
		# code_k_data_both['date'] = pd.to_datetime(code_k_data_both['date'])
		code_k_data_both['date'] = code_k_data_both['date'].map(lambda x: int(x.replace('-', '')))
		code_k_data_sh = code_k_data_both[code_k_data_both['code'] == 'sh.000001'][['date', 'close']]
		code_k_data_sh.columns = ["date", "sh_close"]
		code_k_data_sz = code_k_data_both[code_k_data_both['code'] == 'sz.399001'][['date', 'close']]
		code_k_data_sz.columns = ["date", "sz_close"]
		code_k_data_cy = code_k_data_both[code_k_data_both['code'] == 'sz.399006'][['date', 'close']]
		code_k_data_cy.columns = ["date", "cy_close"]

		code_k_data_both = pd.merge(code_k_data_both, code_k_data_sh, how="left", left_on=['date'], right_on=['date'])
		code_k_data_both = pd.merge(code_k_data_both, code_k_data_sz, how="left", left_on=['date'], right_on=['date'])
		code_k_data_both = pd.merge(code_k_data_both, code_k_data_cy, how="left", left_on=['date'], right_on=['date'])
		# code_k_data_both.dropna(axis=0, inplace=True)

		code_k_data_both = code_k_data_both.sort_values(['code', 'date'])
		code_k_data_both['date_new'] = code_k_data_both['date']
		code_k_data_both = code_k_data_both.groupby('code').apply(lambda x: x.set_index('date'))
		code_k_data_both['close_7d_max'] = code_k_data_both['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=7, window=7, center=False).max())
		code_k_data_both['close_7d_min'] = code_k_data_both['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=7, window=7, center=False).min())
		code_k_data_both['sh_close_7d_max'] = code_k_data_both['sh_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=7, window=7, center=False).max())
		code_k_data_both['sh_close_7d_min'] = code_k_data_both['sh_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=7, window=7, center=False).min())
		code_k_data_both['sz_close_7d_max'] = code_k_data_both['sz_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=7, window=7, center=False).max())
		code_k_data_both['sz_close_7d_min'] = code_k_data_both['sz_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=7, window=7, center=False).min())
		code_k_data_both['cy_close_7d_max'] = code_k_data_both['cy_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=7, window=7, center=False).max())
		code_k_data_both['cy_close_7d_min'] = code_k_data_both['cy_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=7, window=7, center=False).min())
		code_k_data_both['close_cur_day'] = code_k_data_both['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=8, window=8, center=False).apply(lambda y: y[0]))
		code_k_data_both['sh_close_cur_day'] = code_k_data_both['sh_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=8, window=8, center=False).apply(lambda y: y[0]))
		code_k_data_both['sz_close_cur_day'] = code_k_data_both['sz_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=8, window=8, center=False).apply(lambda y: y[0]))
		code_k_data_both['cy_close_cur_day'] = code_k_data_both['cy_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=8, window=8, center=False).apply(lambda y: y[0]))
		code_k_data_both['date_7'] = code_k_data_both['date_new'].groupby(level=0).apply(lambda x: x.rolling(min_periods=8, window=8, center=False).apply(lambda y: y[0]))

		code_k_data_both['pctChg_7_max'] = (code_k_data_both['close_7d_max'] - code_k_data_both['close_cur_day'])/code_k_data_both['close_cur_day']
		code_k_data_both['sh_pctChg_7_max'] = (code_k_data_both['sh_close_7d_max'] - code_k_data_both['sh_close_cur_day'])/code_k_data_both['sh_close_cur_day']
		code_k_data_both['sz_pctChg_7_max'] = (code_k_data_both['sz_close_7d_max'] - code_k_data_both['sz_close_cur_day'])/code_k_data_both['sz_close_cur_day']
		code_k_data_both['cy_pctChg_7_max'] = (code_k_data_both['cy_close_7d_max'] - code_k_data_both['cy_close_cur_day'])/code_k_data_both['cy_close_cur_day']

		code_k_data_both['pctChg_7'] = (code_k_data_both['close'] - code_k_data_both['close_cur_day'])/code_k_data_both['close_cur_day']
		code_k_data_both['sh_pctChg_7'] = (code_k_data_both['sh_close'] - code_k_data_both['sh_close_cur_day'])/code_k_data_both['sh_close_cur_day']
		code_k_data_both['sz_pctChg_7'] = (code_k_data_both['sz_close'] - code_k_data_both['sz_close_cur_day'])/code_k_data_both['sz_close_cur_day']
		code_k_data_both['cy_pctChg_7'] = (code_k_data_both['cy_close'] - code_k_data_both['cy_close_cur_day'])/code_k_data_both['cy_close_cur_day']

		label_7 = code_k_data_both[['date_7', 'pctChg_7', 'sh_pctChg_7', 'sz_pctChg_7','cy_pctChg_7', 'pctChg_7_max', 'sh_pctChg_7_max', 'sz_pctChg_7_max','cy_pctChg_7_max']]
		label_7.columns = ['date', 'pctChg_7', 'sh_pctChg_7', 'sz_pctChg_7','cy_pctChg_7', 'pctChg_7_max', 'sh_pctChg_7_max', 'sz_pctChg_7_max','cy_pctChg_7_max']

		code_k_data_both['close_15d_max'] = code_k_data_both['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=15, window=15, center=False).max())
		code_k_data_both['close_15d_min'] = code_k_data_both['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=15, window=15, center=False).min())
		code_k_data_both['sh_close_15d_max'] = code_k_data_both['sh_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=15, window=15, center=False).max())
		code_k_data_both['sh_close_15d_min'] = code_k_data_both['sh_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=15, window=15, center=False).min())
		code_k_data_both['sz_close_15d_max'] = code_k_data_both['sz_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=15, window=15, center=False).max())
		code_k_data_both['sz_close_15d_min'] = code_k_data_both['sz_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=15, window=15, center=False).min())
		code_k_data_both['cy_close_15d_max'] = code_k_data_both['cy_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=15, window=15, center=False).max())
		code_k_data_both['cy_close_15d_min'] = code_k_data_both['cy_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=15, window=15, center=False).min())
		code_k_data_both['close_cur_day'] = code_k_data_both['close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=16, window=16, center=False).apply(lambda y: y[0]))
		code_k_data_both['sh_close_cur_day'] = code_k_data_both['sh_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=16, window=16, center=False).apply(lambda y: y[0]))
		code_k_data_both['sz_close_cur_day'] = code_k_data_both['sz_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=16, window=16, center=False).apply(lambda y: y[0]))
		code_k_data_both['cy_close_cur_day'] = code_k_data_both['cy_close'].groupby(level=0).apply(lambda x: x.rolling(min_periods=16, window=16, center=False).apply(lambda y: y[0]))
		code_k_data_both['date_15'] = code_k_data_both['date_new'].groupby(level=0).apply(lambda x: x.rolling(min_periods=16, window=16, center=False).apply(lambda y: y[0]))

		code_k_data_both['pctChg_15_max'] = (code_k_data_both['close_15d_max'] - code_k_data_both['close_cur_day'])/code_k_data_both['close_cur_day']
		code_k_data_both['sh_pctChg_15_max'] = (code_k_data_both['sh_close_15d_max'] - code_k_data_both['sh_close_cur_day'])/code_k_data_both['sh_close_cur_day']
		code_k_data_both['sz_pctChg_15_max'] = (code_k_data_both['sz_close_15d_max'] - code_k_data_both['sz_close_cur_day'])/code_k_data_both['sz_close_cur_day']
		code_k_data_both['cy_pctChg_15_max'] = (code_k_data_both['cy_close_15d_max'] - code_k_data_both['cy_close_cur_day'])/code_k_data_both['cy_close_cur_day']
		code_k_data_both['pctChg_15'] = (code_k_data_both['close'] - code_k_data_both['close_cur_day'])/code_k_data_both['close_cur_day']
		code_k_data_both['sh_pctChg_15'] = (code_k_data_both['sh_close'] - code_k_data_both['sh_close_cur_day'])/code_k_data_both['sh_close_cur_day']
		code_k_data_both['sz_pctChg_15'] = (code_k_data_both['sz_close'] - code_k_data_both['sz_close_cur_day'])/code_k_data_both['sz_close_cur_day']
		code_k_data_both['cy_pctChg_15'] = (code_k_data_both['cy_close'] - code_k_data_both['cy_close_cur_day'])/code_k_data_both['cy_close_cur_day']

		label_15 = code_k_data_both[['date_15', 'pctChg_15', 'sh_pctChg_15', 'sz_pctChg_15','cy_pctChg_15', 'pctChg_15_max', 'sh_pctChg_15_max', 'sz_pctChg_15_max','cy_pctChg_15_max']]
		label_15.columns = ['date', 'pctChg_15', 'sh_pctChg_15', 'sz_pctChg_15','cy_pctChg_15', 'pctChg_15_max', 'sh_pctChg_15_max', 'sz_pctChg_15_max','cy_pctChg_15_max']
		# label_7.dropna(axis=0, inplace=True)
		# label_15.dropna(axis=0, inplace=True)
		label_7 = label_7[label_7['date'] < (self.year + 1)*10000]
		label_15 = label_15[label_15['date'] < (self.year + 1)*10000]
		label_7 = label_7.reset_index(level=0, drop=False)
		label_7 = label_7.reset_index(level=0, drop=True)
		label_15 = label_15.reset_index(level=0, drop=False)
		label_15 = label_15.reset_index(level=0, drop=True)
		label = pd.merge(label_7, label_15, how="left", left_on=['code', 'date'], right_on=['code', 'date'])
		# label.dropna(axis=0, inplace=True)
		return label

if __name__ == '__main__':
	path = 'E:/pythonProject/future/data/datafile/raw_feature/code_k_data_v4_'
	year = 2007
	Label = Label(path + str(year) + '.csv', path + str(year + 1) + '.csv', year)
	years = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
	for year in years:
		Label.year = year
		Label.path1 = path + str(year) + '.csv'
		Label.path2 = path + str(year + 1) + '.csv'
		label_raw = Label.get_raw_label()
		label_raw.to_csv('E:/pythonProject/future/data/datafile/label/{year}_raw_v4.csv'.format(year=str(year)),
		                   mode='w', header=True, index=False)



