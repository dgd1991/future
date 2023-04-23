import gc
import time

import tensorflow as tf
import logging
import pandas as pd
import os
import json
from datetime import datetime
# from tensorflow.contrib import layers
# from tensorflow import feature_column
from tensorflow.python.training import queue_runner_impl
from tensorflow.core.util.event_pb2 import SessionLog
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class DeepFM(object):
    def __init__(self, year, train_stype, model_name, saved_model, evaluate_feature='', is_shuffle=True, date=''):
        # for index in range(10):
        #     self.train_data.append('E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(
        #         model_name=model_name, year=str(index + 2008)))
        if year == 2022:
            self.year_test = year
        else:
            self.year_test = year + 1
        self.date = date

        self.evaluate_feature = evaluate_feature
        self.task_type = train_stype
        if is_shuffle:
            self.train_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/shuffled_train_sample_{year}.csv'.format(model_name=model_name, year=str(year))
        else:
            self.train_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name=model_name, year=str(year))
        if self.task_type in ('train', 'evaluate'):
            self.test_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name=model_name, year=str(self.year_test))
        else:
            self.test_data = 'E:/pythonProject/future/data/datafile/prediction_sample/{model_name}/prediction_sample_{date}.csv'.format(model_name=model_name, date=str(self.date))

        # if self.task_type == 'evaluate':
        #     self.test_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/eval_sample_{year}.csv'.format(model_name=model_name, year=str(self.year_test))
        # else:
        #     self.test_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name=model_name, year=str(self.year_test))
        if self.date != "":
            self.prediction_result = 'E:/pythonProject/future/data/datafile/prediction_result/{model_name}/prediction_result_{year}.csv'.format(model_name=saved_model, year=str(self.date))
        else:
            self.prediction_result = 'E:/pythonProject/future/data/datafile/prediction_result/{model_name}/prediction_result_{year}.csv'.format(model_name=saved_model, year=str(self.year_test))
        self.evaluate_result = 'E:/pythonProject/future/data/datafile/prediction_result/{model_name}/evaluate_result_{year}.csv'.format(model_name=saved_model, year=str(self.year_test))
        self.checkpoint_path = "E:\\pythonProject\\future\\saved_model\\{model_name}".format(model_name=str(saved_model))
        self.save_summary_steps = 100000
        self.save_checkpoint_and_eval_step = 100000
        self.every_n_steps = 100000
        self.max_train_step = 5000000
        self.embedding_dim = 10
        self.batch_size = 256
        # self.batch_size = 32
        self.feature_columns_dict = {'code_feas': [], 'cr_trend_field': [], 'industry3_feas': []}
        # self.lr = 0.000005
        self.lr = 0.0001
        self.optimizer = 'Adam'
        self.stddev = 0.1
        self.label_cnt = 6
        self.train_epoch = 1

        self.col_name = ['date', 'code', 'industry_name_level1', 'industry_name_level2', 'industry_name_level3', 'industry_id_level1', 'industry_id_level2', 'industry_id_level3', 'open_ratio', 'close_ratio', 'high_ratio', 'low_ratio', 'pctChg', 'code_market', 'open_ratio_7d_avg', 'close_ratio_7d_avg', 'high_ratio_7d_avg', 'low_ratio_7d_avg', 'market_value', 'peTTM', 'pcfNcfTTM', 'pbMRQ', 'isST', 'turn', 'rsv_3', 'k_value_3', 'd_value_3', 'j_value_3', 'rsv_5', 'k_value_5', 'd_value_5', 'j_value_5', 'rsv_9', 'k_value_9', 'd_value_9', 'j_value_9', 'rsv_19', 'k_value_19', 'd_value_19', 'j_value_19', 'rsv_73', 'k_value_73', 'd_value_73', 'j_value_73', 'macd_positive', 'macd_dif_ratio', 'macd_dif_2', 'macd_dea_2', 'macd_2', 'macd_positive_ratio_2', 'macd_dif_3', 'macd_dea_3', 'macd_3', 'macd_positive_ratio_3', 'macd_dif_5', 'macd_dea_5', 'macd_5', 'macd_positive_ratio_5', 'macd_dif_10', 'macd_dea_10', 'macd_10', 'macd_positive_ratio_10', 'macd_dif_20', 'macd_dea_20', 'macd_20', 'macd_positive_ratio_20', 'macd_dif_40', 'macd_dea_40', 'macd_40', 'macd_positive_ratio_40', 'macd_dif_dea', 'width_2', 'close_mb2_diff', 'width_3', 'close_mb3_diff', 'width_5', 'close_mb5_diff', 'width_10', 'close_mb10_diff', 'width_20', 'close_mb20_diff', 'width_40', 'close_mb40_diff', 'cr_bias_26d', 'cr_26d', 'cr_trend_26d', 'cr_trend_26d_0', 'cr_trend_26d_1', 'cr_trend_26d_2', 'cr_trend_26d_3', 'rsi_3d', 'rsi_5d', 'rsi_10d', 'rsi_20d', 'rsi_40d', 'turn_3d_avg', 'turn_rank_3d', 'turn_3davg_dif', 'turn_3dmax_dif', 'turn_3dmin_dif', 'close_3davg_dif', 'close_3dmax_dif', 'close_3dmin_dif', 'close_3d_dif', 'turn_5d_avg', 'turn_rank_5d', 'turn_5davg_dif', 'turn_5dmax_dif', 'turn_5dmin_dif', 'turn_3_5d_avg', 'turn_3_5dmax_dif', 'turn_3_5dmin_dif', 'close_5davg_dif', 'close_5dmax_dif', 'close_5dmin_dif', 'close_5d_dif', 'close_3_5d_avg', 'turn_10d_avg', 'turn_rank_10d', 'turn_10davg_dif', 'turn_10dmax_dif', 'turn_10dmin_dif', 'turn_5_10d_avg', 'turn_5_10dmax_dif', 'turn_5_10dmin_dif', 'close_10davg_dif', 'close_10dmax_dif', 'close_10dmin_dif', 'close_10d_dif', 'close_5_10d_avg', 'turn_20d_avg', 'turn_rank_20d', 'turn_20davg_dif', 'turn_20dmax_dif', 'turn_20dmin_dif', 'turn_10_20d_avg', 'turn_10_20dmax_dif', 'turn_10_20dmin_dif', 'close_20davg_dif', 'close_20dmax_dif', 'close_20dmin_dif', 'close_20d_dif', 'close_10_20d_avg', 'turn_30d_avg', 'turn_rank_30d', 'turn_30davg_dif', 'turn_30dmax_dif', 'turn_30dmin_dif', 'turn_20_30d_avg', 'turn_20_30dmax_dif', 'turn_20_30dmin_dif', 'close_30davg_dif', 'close_30dmax_dif', 'close_30dmin_dif', 'close_30d_dif', 'close_20_30d_avg', 'turn_60d_avg', 'turn_rank_60d', 'turn_60davg_dif', 'turn_60dmax_dif', 'turn_60dmin_dif', 'turn_30_60d_avg', 'turn_30_60dmax_dif', 'turn_30_60dmin_dif', 'close_60davg_dif', 'close_60dmax_dif', 'close_60dmin_dif', 'close_60d_dif', 'close_30_60d_avg', 'turn_120d_avg', 'turn_rank_120d', 'turn_120davg_dif', 'turn_120dmax_dif', 'turn_120dmin_dif', 'turn_60_120d_avg', 'turn_60_120dmax_dif', 'turn_60_120dmin_dif', 'close_120davg_dif', 'close_120dmax_dif', 'close_120dmin_dif', 'close_120d_dif', 'close_60_120d_avg', 'turn_240d_avg', 'turn_rank_240d', 'turn_240davg_dif', 'turn_240dmax_dif', 'turn_240dmin_dif', 'turn_120_240d_avg', 'turn_120_240dmax_dif', 'turn_120_240dmin_dif', 'close_240davg_dif', 'close_240dmax_dif', 'close_240dmin_dif', 'close_240d_dif', 'close_120_240d_avg', 'max_turn_index3d', 'max_close_index3d', 'min_close_index3d', 'max_turn_close3d', 'max_closs_turn3d', 'min_closs_turn3d', 'max_turn_index5d', 'max_close_index5d', 'min_close_index5d', 'max_turn_close5d', 'max_closs_turn5d', 'min_closs_turn5d', 'max_turn_index10d', 'max_close_index10d', 'min_close_index10d', 'max_turn_close10d', 'max_closs_turn10d', 'min_closs_turn10d', 'max_turn_index20d', 'max_close_index20d', 'min_close_index20d', 'max_turn_close20d', 'max_closs_turn20d', 'min_closs_turn20d', 'max_turn_index30d', 'max_close_index30d', 'min_close_index30d', 'max_turn_close30d', 'max_closs_turn30d', 'min_closs_turn30d', 'max_turn_index60d', 'max_close_index60d', 'min_close_index60d', 'max_turn_close60d', 'max_closs_turn60d', 'min_closs_turn60d', 'max_turn_index120d', 'max_close_index120d', 'min_close_index120d', 'max_turn_close120d', 'max_closs_turn120d', 'min_closs_turn120d', 'max_turn_index240d', 'max_close_index240d', 'min_close_index240d', 'max_turn_close240d', 'max_closs_turn240d', 'min_closs_turn240d', 'pctChg_up_limit_3d', 'pctChg_down_limit_3d', 'pctChg_greater_3_3d', 'pctChg_greater_6_3d', 'pctChg_greater_9_3d', 'pctChg_greater_13_3d', 'pctChg_less_3_3d', 'pctChg_less_6_3d', 'pctChg_less_9_3d', 'pctChg_less_13_3d', 'turn_greater_3_3d', 'turn_greater_6_3d', 'turn_greater_10_3d', 'turn_greater_15_3d', 'turn_greater_21_3d', 'pctChg_up_limit_5d', 'pctChg_down_limit_5d', 'pctChg_greater_3_5d', 'pctChg_greater_6_5d', 'pctChg_greater_9_5d', 'pctChg_greater_13_5d', 'pctChg_less_3_5d', 'pctChg_less_6_5d', 'pctChg_less_9_5d', 'pctChg_less_13_5d', 'turn_greater_3_5d', 'turn_greater_6_5d', 'turn_greater_10_5d', 'turn_greater_15_5d', 'turn_greater_21_5d', 'pctChg_up_limit_10d', 'pctChg_down_limit_10d', 'pctChg_greater_3_10d', 'pctChg_greater_6_10d', 'pctChg_greater_9_10d', 'pctChg_greater_13_10d', 'pctChg_less_3_10d', 'pctChg_less_6_10d', 'pctChg_less_9_10d', 'pctChg_less_13_10d', 'turn_greater_3_10d', 'turn_greater_6_10d', 'turn_greater_10_10d', 'turn_greater_15_10d', 'turn_greater_21_10d', 'pctChg_up_limit_20d', 'pctChg_down_limit_20d', 'pctChg_greater_3_20d', 'pctChg_greater_6_20d', 'pctChg_greater_9_20d', 'pctChg_greater_13_20d', 'pctChg_less_3_20d', 'pctChg_less_6_20d', 'pctChg_less_9_20d', 'pctChg_less_13_20d', 'turn_greater_3_20d', 'turn_greater_6_20d', 'turn_greater_10_20d', 'turn_greater_15_20d', 'turn_greater_21_20d', 'pctChg_up_limit_30d', 'pctChg_down_limit_30d', 'pctChg_greater_3_30d', 'pctChg_greater_6_30d', 'pctChg_greater_9_30d', 'pctChg_greater_13_30d', 'pctChg_less_3_30d', 'pctChg_less_6_30d', 'pctChg_less_9_30d', 'pctChg_less_13_30d', 'turn_greater_3_30d', 'turn_greater_6_30d', 'turn_greater_10_30d', 'turn_greater_15_30d', 'turn_greater_21_30d', 'pctChg_up_limit_60d', 'pctChg_down_limit_60d', 'pctChg_greater_3_60d', 'pctChg_greater_6_60d', 'pctChg_greater_9_60d', 'pctChg_greater_13_60d', 'pctChg_less_3_60d', 'pctChg_less_6_60d', 'pctChg_less_9_60d', 'pctChg_less_13_60d', 'turn_greater_3_60d', 'turn_greater_6_60d', 'turn_greater_10_60d', 'turn_greater_15_60d', 'turn_greater_21_60d', 'pctChg_up_limit_120d', 'pctChg_down_limit_120d', 'pctChg_greater_3_120d', 'pctChg_greater_6_120d', 'pctChg_greater_9_120d', 'pctChg_greater_13_120d', 'pctChg_less_3_120d', 'pctChg_less_6_120d', 'pctChg_less_9_120d', 'pctChg_less_13_120d', 'turn_greater_3_120d', 'turn_greater_6_120d', 'turn_greater_10_120d', 'turn_greater_15_120d', 'turn_greater_21_120d', 'pctChg_3d', 'pctChg_5d', 'pctChg_10d', 'pctChg_20d', 'pctChg_30d', 'pctChg_60d', 'pctChg_120d', 'pctChg_rank_ratio3d', 'pctChg_rank_ratio_industry1_3d', 'pctChg_rank_ratio_industry2_3d', 'pctChg_rank_ratio_industry3_3d', 'turn_rank_industry1_3d', 'turn_rank_industry2_3d', 'turn_rank_industry3_3d', 'pctChg_rank_ratio5d', 'pctChg_rank_ratio_industry1_5d', 'pctChg_rank_ratio_industry2_5d', 'pctChg_rank_ratio_industry3_5d', 'turn_rank_industry1_5d', 'turn_rank_industry2_5d', 'turn_rank_industry3_5d', 'pctChg_rank_ratio10d', 'pctChg_rank_ratio_industry1_10d', 'pctChg_rank_ratio_industry2_10d', 'pctChg_rank_ratio_industry3_10d', 'turn_rank_industry1_10d', 'turn_rank_industry2_10d', 'turn_rank_industry3_10d', 'pctChg_rank_ratio20d', 'pctChg_rank_ratio_industry1_20d', 'pctChg_rank_ratio_industry2_20d', 'pctChg_rank_ratio_industry3_20d', 'turn_rank_industry1_20d', 'turn_rank_industry2_20d', 'turn_rank_industry3_20d', 'pctChg_rank_ratio30d', 'pctChg_rank_ratio_industry1_30d', 'pctChg_rank_ratio_industry2_30d', 'pctChg_rank_ratio_industry3_30d', 'turn_rank_industry1_30d', 'turn_rank_industry2_30d', 'turn_rank_industry3_30d', 'pctChg_rank_ratio60d', 'pctChg_rank_ratio_industry1_60d', 'pctChg_rank_ratio_industry2_60d', 'pctChg_rank_ratio_industry3_60d', 'turn_rank_industry1_60d', 'turn_rank_industry2_60d', 'turn_rank_industry3_60d', 'pctChg_rank_ratio120d', 'pctChg_rank_ratio_industry1_120d', 'pctChg_rank_ratio_industry2_120d', 'pctChg_rank_ratio_industry3_120d', 'turn_rank_industry1_120d', 'turn_rank_industry2_120d', 'turn_rank_industry3_120d', 'industry_id_level3_open_ratio', 'industry_id_level3_close_ratio', 'industry_id_level3_high_ratio', 'industry_id_level3_low_ratio', 'industry_id_level3_pctChg', 'industry_id_level3_rise_ratio', 'industry_id_level3_close', 'industry_id_level3_preclose', 'industry_id_level3_open', 'industry_id_level3_high', 'industry_id_level3_low', 'industry_id_level3_open_ratio_7d_avg', 'industry_id_level3_close_ratio_7d_avg', 'industry_id_level3_high_ratio_7d_avg', 'industry_id_level3_low_ratio_7d_avg', 'industry_id_level3_rsv_5', 'industry_id_level3_k_value_5', 'industry_id_level3_d_value_5', 'industry_id_level3_j_value_5', 'industry_id_level3_rsv_9', 'industry_id_level3_k_value_9', 'industry_id_level3_d_value_9', 'industry_id_level3_j_value_9', 'industry_id_level3_rsv_19', 'industry_id_level3_k_value_19', 'industry_id_level3_d_value_19', 'industry_id_level3_j_value_19', 'industry_id_level3_rsv_73', 'industry_id_level3_k_value_73', 'industry_id_level3_d_value_73', 'industry_id_level3_j_value_73', 'industry_id_level3_macd_dif_ratio', 'industry_id_level3_macd_positive_ratio_2', 'industry_id_level3_macd_positive_ratio_3', 'industry_id_level3_macd_positive_ratio_5', 'industry_id_level3_macd_positive_ratio_10', 'industry_id_level3_macd_positive_ratio_20', 'industry_id_level3_macd_positive_ratio_40', 'industry_id_level3_width_2', 'industry_id_level3_close_mb2_diff', 'industry_id_level3_width_3', 'industry_id_level3_close_mb3_diff', 'industry_id_level3_width_5', 'industry_id_level3_close_mb5_diff', 'industry_id_level3_width_10', 'industry_id_level3_close_mb10_diff', 'industry_id_level3_width_20', 'industry_id_level3_close_mb20_diff', 'industry_id_level3_width_40', 'industry_id_level3_close_mb40_diff', 'industry_id_level3_cr_bias_26d', 'industry_id_level3_rsi_3d', 'industry_id_level3_rsi_5d', 'industry_id_level3_rsi_10d', 'industry_id_level3_rsi_20d', 'industry_id_level3_rsi_40d', 'industry_id_level3_turn_3d_avg', 'industry_id_level3_turn_rank_3d_avg', 'industry_id_level3_turn_3dmax_dif', 'industry_id_level3_turn_3dmin_dif', 'industry_id_level3_turn_5d_avg', 'industry_id_level3_turn_rank_5d_avg', 'industry_id_level3_turn_5dmax_dif', 'industry_id_level3_turn_5dmin_dif', 'industry_id_level3_turn_10d_avg', 'industry_id_level3_turn_rank_10d_avg', 'industry_id_level3_turn_10dmax_dif', 'industry_id_level3_turn_10dmin_dif', 'industry_id_level3_turn_20d_avg', 'industry_id_level3_turn_rank_20d_avg', 'industry_id_level3_turn_20dmax_dif', 'industry_id_level3_turn_20dmin_dif', 'industry_id_level3_turn_30d_avg', 'industry_id_level3_turn_rank_30d_avg', 'industry_id_level3_turn_30dmax_dif', 'industry_id_level3_turn_30dmin_dif', 'industry_id_level3_turn_60d_avg', 'industry_id_level3_turn_rank_60d_avg', 'industry_id_level3_turn_60dmax_dif', 'industry_id_level3_turn_60dmin_dif', 'industry_id_level3_turn_120d_avg', 'industry_id_level3_turn_rank_120d_avg', 'industry_id_level3_turn_120dmax_dif', 'industry_id_level3_turn_120dmin_dif', 'industry_id_level3_turn_240d_avg', 'industry_id_level3_turn_rank_240d_avg', 'industry_id_level3_turn_240dmax_dif', 'industry_id_level3_turn_240dmin_dif', 'industry_id_level3_pctChg_3d', 'industry_id_level3_pctChg_5d', 'industry_id_level3_pctChg_10d', 'industry_id_level3_pctChg_20d', 'industry_id_level3_pctChg_30d', 'industry_id_level3_pctChg_60d', 'industry_id_level3_pctChg_120d', 'industry_id_level3_pctChg_240d', 'industry_id_level3_pctChg_rank_ratio3d', 'industry_id_level3_pctChg_rank_ratio5d', 'industry_id_level3_pctChg_rank_ratio10d', 'industry_id_level3_pctChg_rank_ratio20d', 'industry_id_level3_pctChg_rank_ratio30d', 'industry_id_level3_pctChg_rank_ratio60d', 'industry_id_level3_pctChg_rank_ratio120d', 'industry_id_level3_pctChg_rank_ratio240d', 'label_7', 'pctChg_7', 'sh_pctChg_7', 'label_15', 'pctChg_15', 'sh_pctChg_15']
        self.select_col = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.record_defaults = [[0.0], ['mydefault'], ['mydefault'], ['mydefault'], ['mydefault'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        self.fea_config = {'date': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'code': {'bucket': 210, 'dim': 3, 'type': 'string', 'tower': 'code'}, 'industry_name_level1': {'bucket': 210, 'dim': 3, 'type': 'string', 'tower': 'code'}, 'industry_name_level2': {'bucket': 210, 'dim': 3, 'type': 'string', 'tower': 'code'}, 'industry_name_level3': {'bucket': 210, 'dim': 3, 'type': 'string', 'tower': 'code'}, 'industry_id_level1': {'bucket': 210, 'dim': 5, 'type': 'bucketId', 'tower': 'code'}, 'industry_id_level2': {'bucket': 510, 'dim': 5, 'type': 'bucketId', 'tower': 'code'}, 'industry_id_level3': {'bucket': 1010, 'dim': 5, 'type': 'bucketId', 'tower': 'code'}, 'open_ratio': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'close_ratio': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'high_ratio': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'low_ratio': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'code_market': {'bucket': 20, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'open_ratio_7d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'close_ratio_7d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'high_ratio_7d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'low_ratio_7d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'market_value': {'bucket': 40, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'peTTM': {'bucket': 60, 'dim': 3, 'type': 'bucketId', 'tower': 'code'}, 'pcfNcfTTM': {'bucket': 60, 'dim': 3, 'type': 'bucketId', 'tower': 'code'}, 'pbMRQ': {'bucket': 110, 'dim': 3, 'type': 'bucketId', 'tower': 'code'}, 'isST': {'bucket': 13, 'dim': 3, 'type': 'bucketId', 'tower': 'code'}, 'turn': {'bucket': 110, 'dim': 3, 'type': 'bucketId', 'tower': 'code'}, 'rsv_3': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'k_value_3': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'd_value_3': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'j_value_3': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'rsv_5': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'k_value_5': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'd_value_5': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'j_value_5': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'rsv_9': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'k_value_9': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'd_value_9': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'j_value_9': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'rsv_19': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'k_value_19': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'd_value_19': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'j_value_19': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'rsv_73': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'k_value_73': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'd_value_73': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'j_value_73': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'macd_positive': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_dif_ratio': {'bucket': 10, 'dim': 5, 'type': 'float', 'tower': 'code'}, 'macd_dif_2': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_dea_2': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_2': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_positive_ratio_2': {'bucket': 10, 'dim': 5, 'type': 'float', 'tower': 'code'}, 'macd_dif_3': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_dea_3': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_3': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_positive_ratio_3': {'bucket': 10, 'dim': 2, 'type': 'float', 'tower': 'code'}, 'macd_dif_5': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_dea_5': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_5': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_positive_ratio_5': {'bucket': 10, 'dim': 2, 'type': 'float', 'tower': 'code'}, 'macd_dif_10': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_dea_10': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_10': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_positive_ratio_10': {'bucket': 10, 'dim': 2, 'type': 'float', 'tower': 'code'}, 'macd_dif_20': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_dea_20': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_20': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_positive_ratio_20': {'bucket': 10, 'dim': 2, 'type': 'float', 'tower': 'code'}, 'macd_dif_40': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_dea_40': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_40': {'bucket': 12, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'macd_positive_ratio_40': {'bucket': 10, 'dim': 2, 'type': 'float', 'tower': 'code'}, 'macd_dif_dea': {'bucket': 13, 'dim': 2, 'type': 'bucketId', 'tower': 'code'}, 'width_2': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'close_mb2_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'width_3': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'close_mb3_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'width_5': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'close_mb5_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'width_10': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'close_mb10_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'width_20': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'close_mb20_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'width_40': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'close_mb40_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'cr_bias_26d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'cr_26d': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'cr_trend_26d': {'bucket': 750010, 'dim': 10, 'type': 'bucketId', 'tower': 'code', 'shared_field': 'cr_trend_field'}, 'cr_trend_26d_0': {'bucket': 750010, 'dim': 10, 'type': 'bucketId', 'tower': 'code', 'shared_field': 'cr_trend_field'}, 'cr_trend_26d_1': {'bucket': 750010, 'dim': 10, 'type': 'bucketId', 'tower': 'code', 'shared_field': 'cr_trend_field'}, 'cr_trend_26d_2': {'bucket': 750010, 'dim': 10, 'type': 'bucketId', 'tower': 'code', 'shared_field': 'cr_trend_field'}, 'cr_trend_26d_3': {'bucket': 750010, 'dim': 10, 'type': 'bucketId', 'tower': 'code', 'shared_field': 'cr_trend_field'}, 'rsi_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'rsi_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'rsi_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'rsi_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'rsi_40d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_3d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_3d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_3davg_dif': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'turn_3dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_3dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'close_3davg_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_3dmax_dif': {'bucket': 110, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_3dmin_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_3d_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'turn_5d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_5davg_dif': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'turn_5dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_5dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_3_5d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_3_5dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_3_5dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_5davg_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_5dmax_dif': {'bucket': 110, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_5dmin_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_5d_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_3_5d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_10d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_10davg_dif': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'turn_10dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_10dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_5_10d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_5_10dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_5_10dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_10davg_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_10dmax_dif': {'bucket': 110, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_10dmin_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_10d_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_5_10d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_20d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_20davg_dif': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'turn_20dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_20dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_10_20d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_10_20dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_10_20dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_20davg_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_20dmax_dif': {'bucket': 110, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_20dmin_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_20d_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_10_20d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_30d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_30davg_dif': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'turn_30dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_30dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_20_30d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_20_30dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_20_30dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_30davg_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_30dmax_dif': {'bucket': 110, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_30dmin_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_30d_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_20_30d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_60d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_60davg_dif': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'turn_60dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_60dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_30_60d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_30_60dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_30_60dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_60davg_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_60dmax_dif': {'bucket': 110, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_60dmin_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_60d_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_30_60d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_120d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_120davg_dif': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'turn_120dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_120dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_60_120d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_60_120dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_60_120dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_120davg_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_120dmax_dif': {'bucket': 110, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_120dmin_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_120d_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_60_120d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_240d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_240d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_240davg_dif': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'turn_240dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_240dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_120_240d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_120_240dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'turn_120_240dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_240davg_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_240dmax_dif': {'bucket': 110, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'close_240dmin_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_240d_dif': {'bucket': 510, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'close_120_240d_avg': {'bucket': 210, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_index3d': {'bucket': 13, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_close_index3d': {'bucket': 13, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'min_close_index3d': {'bucket': 13, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_close3d': {'bucket': 510, 'dim': 5, 'type': 'float2bucket', 'tower': 'code'}, 'max_closs_turn3d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'min_closs_turn3d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_index5d': {'bucket': 15, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_close_index5d': {'bucket': 15, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'min_close_index5d': {'bucket': 15, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_close5d': {'bucket': 510, 'dim': 5, 'type': 'float2bucket', 'tower': 'code'}, 'max_closs_turn5d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'min_closs_turn5d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_index10d': {'bucket': 20, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_close_index10d': {'bucket': 20, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'min_close_index10d': {'bucket': 20, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_close10d': {'bucket': 510, 'dim': 5, 'type': 'float2bucket', 'tower': 'code'}, 'max_closs_turn10d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'min_closs_turn10d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_index20d': {'bucket': 30, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_close_index20d': {'bucket': 30, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'min_close_index20d': {'bucket': 30, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_close20d': {'bucket': 510, 'dim': 5, 'type': 'float2bucket', 'tower': 'code'}, 'max_closs_turn20d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'min_closs_turn20d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_index30d': {'bucket': 40, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_close_index30d': {'bucket': 40, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'min_close_index30d': {'bucket': 40, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_close30d': {'bucket': 510, 'dim': 5, 'type': 'float2bucket', 'tower': 'code'}, 'max_closs_turn30d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'min_closs_turn30d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_index60d': {'bucket': 70, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_close_index60d': {'bucket': 70, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'min_close_index60d': {'bucket': 70, 'dim': 2, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_close60d': {'bucket': 510, 'dim': 5, 'type': 'float2bucket', 'tower': 'code'}, 'max_closs_turn60d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'min_closs_turn60d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_index120d': {'bucket': 130, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'max_close_index120d': {'bucket': 130, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'min_close_index120d': {'bucket': 130, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_close120d': {'bucket': 510, 'dim': 5, 'type': 'float2bucket', 'tower': 'code'}, 'max_closs_turn120d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'min_closs_turn120d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_index240d': {'bucket': 250, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'max_close_index240d': {'bucket': 250, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'min_close_index240d': {'bucket': 250, 'dim': 3, 'type': 'float2bucket', 'tower': 'code'}, 'max_turn_close240d': {'bucket': 510, 'dim': 5, 'type': 'float2bucket', 'tower': 'code'}, 'max_closs_turn240d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'min_closs_turn240d': {'bucket': 210, 'dim': 4, 'type': 'float2bucket', 'tower': 'code'}, 'pctChg_up_limit_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_down_limit_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_3_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_6_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_9_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_13_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_3_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_6_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_9_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_13_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_3_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_6_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_10_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_15_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_21_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_up_limit_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_down_limit_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_3_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_6_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_9_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_13_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_3_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_6_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_9_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_13_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_3_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_6_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_10_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_15_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_21_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_up_limit_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_down_limit_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_3_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_6_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_9_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_13_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_3_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_6_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_9_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_13_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_3_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_6_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_10_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_15_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_21_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_up_limit_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_down_limit_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_3_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_6_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_9_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_13_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_3_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_6_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_9_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_13_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_3_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_6_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_10_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_15_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_21_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_up_limit_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_down_limit_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_3_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_6_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_9_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_13_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_3_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_6_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_9_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_13_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_3_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_6_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_10_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_15_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_21_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_up_limit_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_down_limit_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_3_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_6_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_9_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_13_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_3_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_6_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_9_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_13_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_3_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_6_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_10_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_15_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_21_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_up_limit_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_down_limit_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_3_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_6_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_9_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_greater_13_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_3_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_6_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_9_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_less_13_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_3_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_6_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_10_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_15_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_greater_21_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_3d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_5d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_10d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_20d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_30d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_60d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_120d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio3d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry1_3d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry2_3d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry3_3d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry1_3d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry2_3d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry3_3d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio5d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry1_5d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry2_5d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry3_5d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry1_5d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry2_5d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry3_5d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio10d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry1_10d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry2_10d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry3_10d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry1_10d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry2_10d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry3_10d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio20d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry1_20d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry2_20d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry3_20d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry1_20d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry2_20d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry3_20d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio30d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry1_30d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry2_30d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry3_30d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry1_30d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry2_30d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry3_30d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio60d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry1_60d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry2_60d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry3_60d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry1_60d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry2_60d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry3_60d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio120d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry1_120d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry2_120d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'pctChg_rank_ratio_industry3_120d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry1_120d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry2_120d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'turn_rank_industry3_120d': {'bucket': 11, 'dim': 3, 'type': 'float', 'tower': 'code'}, 'industry_id_level3_open_ratio': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_close_ratio': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_high_ratio': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_low_ratio': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_rise_ratio': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_close': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_preclose': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_open': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_high': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_low': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_open_ratio_7d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_close_ratio_7d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_high_ratio_7d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_low_ratio_7d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_rsv_5': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_k_value_5': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_d_value_5': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_j_value_5': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_rsv_9': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_k_value_9': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_d_value_9': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_j_value_9': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_rsv_19': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_k_value_19': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_d_value_19': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_j_value_19': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_rsv_73': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_k_value_73': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_d_value_73': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_j_value_73': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_macd_dif_ratio': {'bucket': 10, 'dim': 5, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_macd_positive_ratio_2': {'bucket': 10, 'dim': 5, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_macd_positive_ratio_3': {'bucket': 10, 'dim': 2, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_macd_positive_ratio_5': {'bucket': 10, 'dim': 2, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_macd_positive_ratio_10': {'bucket': 10, 'dim': 2, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_macd_positive_ratio_20': {'bucket': 10, 'dim': 2, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_macd_positive_ratio_40': {'bucket': 10, 'dim': 2, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_width_2': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_close_mb2_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_width_3': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_close_mb3_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_width_5': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_close_mb5_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_width_10': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_close_mb10_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_width_20': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_close_mb20_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_width_40': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_close_mb40_diff': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_cr_bias_26d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_rsi_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_rsi_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_rsi_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_rsi_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_rsi_40d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_3d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_rank_3d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_3dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_3dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_5d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_rank_5d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_5dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_5dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_10d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_rank_10d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_10dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_10dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_20d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_rank_20d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_20dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_20dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_30d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_rank_30d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_30dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_30dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_60d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_rank_60d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_60dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_60dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_120d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_rank_120d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_120dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_120dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_240d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_rank_240d_avg': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_240dmax_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_turn_240dmin_dif': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_240d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_rank_ratio3d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_rank_ratio5d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_rank_ratio10d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_rank_ratio20d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_rank_ratio30d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_rank_ratio60d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_rank_ratio120d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'industry_id_level3_pctChg_rank_ratio240d': {'bucket': 210, 'dim': 3, 'type': 'float', 'tower': 'industry3'}, 'label_7': {'bucket': 10, 'dim': 5, 'type': 'float', 'tower': 'industry3'}, 'pctChg_7': {'bucket': 10, 'dim': 5, 'type': 'float', 'tower': 'industry3'}, 'sh_pctChg_7': {'bucket': 10, 'dim': 5, 'type': 'float', 'tower': 'industry3'}, 'label_15': {'bucket': 10, 'dim': 5, 'type': 'float', 'tower': 'industry3'}, 'pctChg_15': {'bucket': 10, 'dim': 5, 'type': 'float', 'tower': 'industry3'}, 'sh_pctChg_15': {'bucket': 10, 'dim': 5, 'type': 'float', 'tower': 'industry3'}}

        self.get_fea_columns()
        self.dnn_dims, self.industry3_dims = self.init_variable()
        self.fea_count = sum(self.select_col) - self.label_cnt

    def init_variable(self):
        dnn_hidden_1 = 1500
        dnn_hidden_2 = 1024
        dnn_hidden_3 = 700
        dnn_hidden_4 = 512
        dnn_hidden_5 = 256
        dnn_hidden_6 = 128
        dnn_hidden_7 = 64
        dnn_dims = [dnn_hidden_1, dnn_hidden_2, dnn_hidden_3, dnn_hidden_4, dnn_hidden_5, dnn_hidden_6, dnn_hidden_7]

        industry3_hidden_1 = 256
        industry3_hidden_2 = 128
        industry3_hidden_3 = 64
        industry3_dims = [industry3_hidden_1, industry3_hidden_2, industry3_hidden_3]
        return dnn_dims, industry3_dims

    def decode_csv(self, line):
        data = tf.decode_csv(line, record_defaults=self.record_defaults, field_delim=',', use_quote_delim=True, na_value='', name=None)
        label = data[-self.label_cnt:]
        features = {}
        for index in range(len(self.col_name) - self.label_cnt):
            if self.select_col[index] == 1:
                fea_name = self.col_name[index]
                config = self.fea_config[fea_name]
                fea_type = config['type']
                val = data[index]
                if self.task_type=="evaluate" and self.evaluate_feature == fea_name:
                    val = tf.random_shuffle(val)
                if fea_type in ('mydefault'):
                    val = tf.cast(val, dtype=tf.string)
                elif fea_type in ('bucketId', 'float2bucket'):
                    val = tf.cast(val, dtype=tf.int32)
                elif fea_type in ('float'):
                    val = tf.cast(val, dtype=tf.float32)
                key = self.col_name[index]
                features[key] = val

        features['label_7'] = label[0]
        features['pctChg_7'] = label[1]
        features['sh_pctChg_7'] = label[2]
        features['label_15'] = label[3]
        features['pctChg_15'] = label[4]
        features['sh_pctChg_15'] = label[5]

        features['date'] = data[0]
        features['code'] = tf.cast(data[1], dtype=tf.string)
        features['industry_name_level1'] = tf.cast(data[2], dtype=tf.string)
        features['industry_name_level2'] = tf.cast(data[3], dtype=tf.string)
        features['industry_name_level3'] = tf.cast(data[4], dtype=tf.string)
        return features, label

    def train_input_fn_from_csv(self, data_path, epoch=1, batch_size=1024):
        with tf.device('/cpu:0'):
            dataset = tf.data.TextLineDataset(data_path).skip(1)
            # dataset = dataset.repeat(epoch).shuffle(buffer_size=batch_size*10000, seed=None, reshuffle_each_iteration=True).batch(batch_size)
            dataset = dataset.repeat(epoch).batch(batch_size)
            dataset = dataset.map(self.decode_csv, num_parallel_calls=10).prefetch(batch_size*2)
            dataset = dataset.make_one_shot_iterator()
            features_batch, label_batch = dataset.get_next()
            print('=================iterator============')
        return features_batch, label_batch

    def test_input_fn_from_csv(self, data_path, epoch=1, batch_size=1024):
        with tf.device('/cpu:0'):
            dataset = tf.data.TextLineDataset(data_path).skip(1)
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(self.decode_csv, num_parallel_calls=10).prefetch(5)
            dataset = dataset.make_one_shot_iterator()
            features_batch, label_batch = dataset.get_next()
            print('=================iterator============')
        return features_batch, label_batch

    def hash_embedding(self, name, hash_bucket, dim):
        cate_feature = tf.feature_column.categorical_column_with_hash_bucket(name,
                                                                          hash_bucket,
                                                                          dtype=tf.string)
        emb_col = tf.feature_column.embedding_column(
            cate_feature,
            dimension=dim,
            combiner='mean',initializer=tf.truncated_normal_initializer(stddev=self.stddev)
        )
        return emb_col

    # embedding for map
    def map_embedding(self, key_name, val_name, hash_bucket, dim):
        cate_feature = tf.feature_column.categorical_column_with_hash_bucket(key_name,hash_bucket,dtype=tf.string)
        w_cate_feature = tf.feature_column.weighted_categorical_column(cate_feature,val_name,dtype=tf.float32)
        emb_col = tf.feature_column.embedding_column(w_cate_feature, dimension=dim,initializer=tf.truncated_normal_initializer(stddev=self.stddev))
        return emb_col

    # embedding for hashed category
    def index_embedding(self, key_name, hash_bucket, dim):
        id_feature = tf.feature_column.categorical_column_with_identity(key_name, num_buckets=hash_bucket, default_value=0)
        emb_col = tf.feature_column.embedding_column(id_feature, dim, initializer=tf.truncated_normal_initializer(stddev=self.stddev))
        # ind_col = feature_column.indicator_column(id_feature)
        return emb_col
    def numeric_column(self, key_name):
        numeric_feature = tf.feature_column.numeric_column(key_name, shape=(1,), default_value=0, dtype=tf.float32, normalizer_fn=None)
        return numeric_feature

    def shared_embedding(self, key_name, val_name, hash_bucket):
        cate_feature = tf.feature_column.categorical_column_with_hash_bucket(key_name,hash_bucket,dtype=tf.string)
        if val_name != "":
            w_cate_feature = tf.feature_column.weighted_categorical_column(cate_feature,val_name,dtype=tf.float32)
            return w_cate_feature
        return cate_feature
    def shared_bucketId_embedding(self, key_name, hash_bucket):
        id_feature = tf.feature_column.categorical_column_with_identity(key_name, num_buckets=hash_bucket, default_value=0)
        return id_feature

    def get_fea_columns(self):
        for field_idx in range(len(self.col_name) - self.label_cnt):
            if self.select_col[field_idx] == 1:
                fea_name = self.col_name[field_idx]
                config = self.fea_config[fea_name]
                fea_bucket = config['bucket']
                fea_type = config['type']
                fea_tower = config['tower']
                dim = config['dim']
                fea_field = 'other'
                if config.__contains__('shared_field'):
                    fea_field = config['shared_field']
                if fea_field in ('cr_trend_field'):
                    value = self.shared_bucketId_embedding(fea_name, fea_bucket)
                    self.feature_columns_dict['cr_trend_field'].append(value)
                else:
                    if fea_type in ('mydefault'):
                        value = self.hash_embedding(fea_name, fea_bucket, dim)
                    elif fea_type in ('bucketId', 'float2bucket'):
                        value = self.index_embedding(fea_name, fea_bucket, dim)
                    elif fea_type in ('float'):
                        value = self.numeric_column(fea_name)
                    if fea_tower == 'code':
                        self.feature_columns_dict['code_feas'].append(value)
                    elif fea_tower == 'industry3':
                        self.feature_columns_dict['industry3_feas'].append(value)
    def model_fn_params(self):
        model_fn_params_dict = {'lr': self.lr, 'optimizer': self.optimizer, 'feature_columns': self.feature_columns_dict}
        return model_fn_params_dict

    def dnn(self, net, dims, tower):
        for idx, units in enumerate(dims):
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu, name=tower + '_dnn_' + str(idx))
            net = tf.layers.batch_normalization(inputs=net, name=tower + '_concat_bn' + str(idx), reuse=tf.AUTO_REUSE)
        # net = tf.layers.dense(net, units=1)
        return net
    def cross_layer(self, net, cross_layers):
        cross_layer = net
        with tf.name_scope('cross'):
            for idx in range(cross_layers):
                cross_layer = tf.layers.dense(cross_layer, units=1, name='cross_layer_' + str(idx))
                cross_layer = net * cross_layer + cross_layer
        return cross_layer

    def model_fn(self, features, labels, mode, params):
        #
        code_feas = params['feature_columns']['code_feas']
        industry3_feas = params['feature_columns']['industry3_feas']
        cr_trend_field = params['feature_columns']['cr_trend_field']

        with tf.device('/cpu:0'):
            code_feas = code_feas + tf.feature_column.shared_embedding_columns(cr_trend_field, 8, combiner='mean', initializer=tf.truncated_normal_initializer(self.stddev), shared_embedding_collection_name='cr_trend_field')
            code_embed_input = tf.feature_column.input_layer(features, code_feas)
            industry3_embed_input = tf.feature_column.input_layer(features, industry3_feas)
        with tf.device('/gpu:0'):
            dnn = self.dnn(code_embed_input, self.dnn_dims, 'code')
            cross = self.cross_layer(code_embed_input, 6)
            industry3_dnn = self.dnn(industry3_embed_input, self.industry3_dims, 'industry3')
            dcn = tf.concat([dnn, cross], axis=1)
            y_ctr = tf.layers.dense(dcn, units=1)

        ctr_pred = tf.sigmoid(y_ctr, name="prediction")
        ctr_pred = tf.reshape(ctr_pred, [-1])
        # ctr_pred = tf.reshape(y_ctr, [-1])
        # predict
        # label_1d, label_3d, label_5d, label_7d
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'predition': ctr_pred,
                'label_7': features['label_7'],
                'pctChg_7': features['pctChg_7'],
                'sh_pctChg_7': features['sh_pctChg_7'],
                'label_15': features['label_15'],
                'pctChg_15': features['pctChg_15'],
                'sh_pctChg_15': features['sh_pctChg_15'],
                'code': features['code'],
                'date': features['date'],
                'industry_name_level1': features['industry_name_level1'],
                'industry_name_level2': features['industry_name_level2'],
                'industry_name_level3': features['industry_name_level3']
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        # label_7
        ground_truth_ctr = tf.reshape(labels[0, :], [-1])
        # ground_truth_ctr_weight = tf.reshape(labels[2, :], [-1])

        # # label_15
        # ground_truth_ctr = tf.reshape(labels[6, :], [-1])
        # ground_truth_ctr_weight = tf.reshape(labels[8, :], [-1])

        loss = tf.reduce_mean(tf.losses.log_loss(labels=ground_truth_ctr, predictions=ctr_pred))

        # eval
        auc_ctr = tf.metrics.auc(labels=ground_truth_ctr, predictions=ctr_pred, name='auc_ctr_op')
        metrics = {'auc_ctr': auc_ctr}
        tf.summary.scalar('auc_ctr', auc_ctr[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        # train
        if params['optimizer'] == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'], beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif params['optimizer'] == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['lr'], initial_accumulator_value=1e-8)
        elif params['optimizer'] == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=params['lr'], momentum=0.95)
        elif params['optimizer'] == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(params['lr'])
        else:
            optimizer = tf.train.GradientDescentOptimizer(params['lr'])

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        # logging_hook = tf.train.LoggingTensorHook({'loss': loss, 'ctr_pred': tf.reduce_mean(ctr_pred), 'ground_truth_ctr': tf.reduce_mean(ground_truth_ctr), 'loss': loss}, every_n_iter=1)
        # return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics, training_hooks=[logging_hook])
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

    def run(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        session_config = tf.ConfigProto(
                                        # device_count={'GPU': 0},
                                        # log_device_placement=True,
                                        inter_op_parallelism_threads=0,
                                        intra_op_parallelism_threads=0,
                                        allow_soft_placement=True)
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        session_config.allow_soft_placement = True
        session_config.gpu_options.allow_growth = True

        classifier = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params=self.model_fn_params(),
            config=tf.estimator.RunConfig(
                session_config=session_config,
                model_dir=self.checkpoint_path,
                tf_random_seed=2020,
                save_summary_steps=self.save_summary_steps,
                save_checkpoints_steps=self.save_checkpoint_and_eval_step,
                keep_checkpoint_max=10)
        )

        if self.task_type == 'train':
            print("......................Start training......................")
            hooks = []
            train_spec = tf.estimator.TrainSpec(
                input_fn=lambda: self.train_input_fn_from_csv(data_path=self.train_data, epoch=self.train_epoch, batch_size=self.batch_size),
                max_steps=self.max_train_step)
            eval_spec = tf.estimator.EvalSpec(
                input_fn=lambda: self.train_input_fn_from_csv(data_path=self.test_data, epoch=1, batch_size=20000),throttle_secs=300)

            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

        elif self.task_type == 'evaluate':
            evaluate_result = classifier.evaluate(input_fn=lambda: self.train_input_fn_from_csv(data_path=self.test_data, epoch=1, batch_size=self.batch_size))
            print(evaluate_result)
            # result = pd.DataFrame({self.evaluate_feature: list([self.evaluate_feature]),
            #                         'auc_ctr': list([evaluate_result['auc_ctr']]),
            #                        'loss': list([evaluate_result['loss']]),
            #                        'global_step': list([evaluate_result['global_step']])
            #                        })
            # result.to_csv(self.evaluate_result, mode='a', header=False, index=False, encoding='utf-8')
        elif self.task_type == 'predict':
            print("......................Start predict......................")
            predictions = classifier.predict(
                input_fn=lambda: self.test_input_fn_from_csv(data_path=self.test_data, epoch=1,
                                                              batch_size=self.batch_size),
                predict_keys=None,
                checkpoint_path=None,
                yield_single_examples=False
            )
            try:
                while (True):
                    predict_result = next(predictions)
                    result = pd.DataFrame({'date': list(predict_result['date']),
                                           'code': list(predict_result['code']),
                                           'industry_name_level1': list(predict_result['industry_name_level1']),
                                           'industry_name_level2': list(predict_result['industry_name_level2']),
                                           'industry_name_level3': list(predict_result['industry_name_level3']),
                                           'predition': list(predict_result['predition']),
                                           'label_7': list(predict_result['label_7']),
                                           'pctChg_7': list(predict_result['pctChg_7']),
                                           'sh_pctChg_7': list(predict_result['sh_pctChg_7']),
                                           'label_15': list(predict_result['label_15']),
                                           'pctChg_15': list(predict_result['pctChg_15']),
                                           'sh_pctChg_15': list(predict_result['sh_pctChg_15'])
                                           })
                    result.to_csv(self.prediction_result, mode='a',header=False, index=False, encoding='utf-8')
                    # predict_result = predict_result['predition_ctr']
                    # predict_result = predict_result['y_ctr']
                    # predict_result = predict_result['ground_truth_ctr1']
                    # predict_result = predict_result['loss']
                    # print("==================")
                    # print(predict_result['predition_ctr'])
                    # print(predict_result['label'])
                    # # print(predict_result['code'])
                    # print("==================")
            except StopIteration:
                print(
                    "predict finish..., idx: {0}, time consume: {1}".format(1, 1))

        # elif self.task_type == 'savemodel':
        #     if self.is_chief:
        #         print("......................Start savemodel......................")
        #         classifier.export_savedmodel(self.output_dir, self.serving_input_receiver_fn, strip_default_attrs=True)


#
def main():
    model = DeepFM(2010, 'predict', 'model_v6')
    test_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}_test.csv'.format(model_name='model_v5', year=str(2010))
    features_batch, label_batch = model.test_input_fn_from_csv(data_path=test_data, epoch=1, batch_size=1)

    # dataset = model.test_input_fn_from_csv(data_path=test_data, epoch=1, batch_size=2)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        # print(sess.run([data]))
        # print(sess.run([label_batch]))
        # print(sess.run([features_batch, label_batch]))
        # print(sess.run(dataset))
        while True:
            features_batch, label_batch = sess.run([features_batch, label_batch])
            print(features_batch)
            # dataset = sess.run([dataset])
            # print(dataset)
        # print(sess.run([features_batch, label_batch]))
        # print(sess.run([features_batch, label_batch]))
        # print(sess.run([features_batch, label_batch]))
        # print(sess.run([features_batch, label_batch]))
        # print(sess.run([features_batch, label_batch]))

        # coord.request_stop()
        # coord.join(threads)
        print('start')
        print('end')

if __name__ == "__main__":
    # rm -r E:/pythonProject/future/saved_model/*
    # main()
    # tf.test.is_gpu_available()
    # years = [2011, 2012, 2013, 2014]
    # for year in years:
    #     model = DeepFM(year, 'train', 'model_v2')
    #     model.run()

    predict_date = '2023-02-16'
    years = [2021,2022]
    # years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,2021,2022]
    years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,2021,2022]

    # years = [2023]
    model_name = 'model_v16'
    saved_model_name = 'saved_model_v16_test1'
    is_shuffle = True
    epoch = 1
    for year in years:
        for epo in range(epoch):
            train_data_path_raw = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name=model_name, year=str(year))
            train_data_path_raw_last = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name=model_name, year=str(year-1))
            train_data_path = 'E:/pythonProject/future/data/datafile/sample/{model_name}/shuffled_train_sample_{year}.csv'.format(model_name=model_name, year=str(year))
            if not os.path.isfile(train_data_path):
                train_data_raw = pd.read_csv(train_data_path_raw).sample(frac=1).round(5)
                train_data_raw.to_csv(train_data_path, mode='w', header=True, index=False, encoding='utf-8')
                del train_data_raw
                gc.collect()
            model = DeepFM(year, 'train', model_name, saved_model_name, evaluate_feature='', is_shuffle=is_shuffle, date="")
            model.lr = 0.0001
            model.run()

            # model = DeepFM(year, 'predict', model_name, saved_model_name, evaluate_feature='', is_shuffle=is_shuffle, date=predict_date)
            # if os.path.isfile(model.prediction_result):
            #     os.remove(model.prediction_result)
            # model.run()

