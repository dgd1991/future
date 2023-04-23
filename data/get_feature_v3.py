import pandas as pd

turn_diff_down = 2
turn_diff_up = 30
close_diff_down = 1.1
close_diff_up = 1.2
amount_up = 100.5
amount_down = 5

raw_k_data = pd.read_csv('E:/pythonProject/future/data/datafile/prediction_sample/model_v12/test_2023.csv')
raw_k_data = raw_k_data[(raw_k_data['turn_diff']>turn_diff_down) & (raw_k_data['turn_diff']<turn_diff_up) & (raw_k_data['close_diff']<close_diff_up) & (raw_k_data['close_diff']>close_diff_down) & (raw_k_data['amount']>amount_down) & (raw_k_data['amount']<amount_up)]

print(raw_k_data[['date','code','turn','close','amount','pctChg_7']])

print(raw_k_data['pctChg_7'].mean())
