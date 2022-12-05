industry_id_level3_k_data_out = industry_id_level3_k_data[['industry_id_level3_open', 'industry_id_level3_close', 'industry_id_level3_high', 'industry_id_level3_low']]
industry_id_level3_k_data_out = industry_id_level3_k_data_out.reset_index(level=0, drop=True)
industry_id_level3_k_data_out = industry_id_level3_k_data_out.reset_index(level=0, drop=False)
industry_id_level3_k_data_out.to_csv('E:/pythonProject/future/data/datafile/industry/' + 'industry_id_level3_' + str(year) + '.csv', mode='w', header=True, index=False)
del industry_id_level3_k_data_out
gc.collect()