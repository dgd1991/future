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
    def __init__(self, year, train_stype, model_name):
        # self.train_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}'.format(model_name=model_name)
        # self.test_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name='tmp', year=str(2021))
        self.train_data = []
        self.train_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name=model_name, year=str(year))
        # for index in range(10):
        #     self.train_data.append('E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(
        #         model_name=model_name, year=str(index + 2008)))

        # self.train_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name=model_name, year=str(year))
        # self.test_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name=model_name, year=str(year+1))
        self.test_data = 'E:/pythonProject/future/data/datafile/sample/{model_name}/train_sample_{year}.csv'.format(model_name=model_name, year=str(year+1))
        self.prediction_result = 'E:/pythonProject/future/data/datafile/prediction_result/{model_name}/prediction_result_{year}.csv'.format(model_name=model_name, year=str(year+1))
        self.task_type = train_stype
        # self.checkpoint_path = "E:/pythonProject/future/saved_model"
        self.checkpoint_path = "E:\\pythonProject\\future\\saved_model\\{model_name}".format(model_name=str(model_name))
        self.save_summary_steps = 100000
        self.save_checkpoint_and_eval_step = 100000
        self.every_n_steps = 100000
        self.max_train_step = 5000000
        self.embedding_dim = 10
        self.batch_size = 256
        self.feature_columns_dict = {'other_feas': []}
        self.lr = 0.00001
        self.optimizer = 'Adam'
        self.stddev = 0.1
        self.label_cnt = 8
        # self.col_name = ['code', 'open_ratio', 'high_ratio', 'low_ratio', 'turn', 'preclose', 'amount', 'pctChg', 'peTTM', 'pcfNcfTTM', 'pbMRQ', 'isST', 'open_ratio_1d', 'high_ratio_1d', 'low_ratio_1d', 'turn_1d', 'pctChg_1d', 'open_ratio_2d', 'high_ratio_2d', 'low_ratio_2d', 'turn_2d', 'pctChg_2d', 'open_ratio_3d', 'high_ratio_3d', 'low_ratio_3d', 'turn_3d', 'pctChg_3d', 'open_ratio_4d', 'high_ratio_4d', 'low_ratio_4d', 'turn_4d', 'pctChg_4d', 'open_ratio_7d', 'high_ratio_7d', 'low_ratio_7d', 'turn_7d', 'pctChg_7d', 'open_ratio_10d', 'high_ratio_10d', 'low_ratio_10d', 'turn_10d', 'pctChg_10d', 'open_ratio_13d', 'high_ratio_13d', 'low_ratio_13d', 'turn_13d', 'pctChg_13d', 'open_ratio_16d', 'high_ratio_16d', 'low_ratio_16d', 'turn_16d', 'pctChg_16d', 'open_ratio_19d', 'high_ratio_19d', 'low_ratio_19d', 'turn_19d', 'pctChg_19d', 'open_ratio_22d', 'high_ratio_22d', 'low_ratio_22d', 'turn_22d', 'pctChg_22d', 'open_ratio_25d', 'high_ratio_25d', 'low_ratio_25d', 'turn_25d', 'pctChg_25d', 'open_ratio_28d', 'high_ratio_28d', 'low_ratio_28d', 'turn_28d', 'pctChg_28d', 'open_ratio_31d', 'high_ratio_31d', 'low_ratio_31d', 'turn_31d', 'pctChg_31d', 'open_ratio_34d', 'high_ratio_34d', 'low_ratio_34d', 'turn_34d', 'pctChg_34d', 'open_ratio_37d', 'high_ratio_37d', 'low_ratio_37d', 'turn_37d', 'pctChg_37d', 'open_ratio_40d', 'high_ratio_40d', 'low_ratio_40d', 'turn_40d', 'pctChg_40d', 'open_ratio_43d', 'high_ratio_43d', 'low_ratio_43d', 'turn_43d', 'pctChg_43d', 'open_ratio_46d', 'high_ratio_46d', 'low_ratio_46d', 'turn_46d', 'pctChg_46d', 'open_ratio_49d', 'high_ratio_49d', 'low_ratio_49d', 'turn_49d', 'pctChg_49d', 'open_ratio_52d', 'high_ratio_52d', 'low_ratio_52d', 'turn_52d', 'pctChg_52d', 'open_ratio_55d', 'high_ratio_55d', 'low_ratio_55d', 'turn_55d', 'pctChg_55d', 'open_ratio_58d', 'high_ratio_58d', 'low_ratio_58d', 'turn_58d', 'pctChg_58d', 'open_ratio_61d', 'high_ratio_61d', 'low_ratio_61d', 'turn_61d', 'pctChg_61d', 'open_ratio_64d', 'high_ratio_64d', 'low_ratio_64d', 'turn_64d', 'pctChg_64d', 'industry', 'roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare', 'NRTurnRatio', 'INVTurnRatio', 'CATurnRatio', 'AssetTurnRatio', 'YOYEquity', 'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI', 'currentRatio', 'quickRatio', 'cashRatio', 'YOYLiability', 'liabilityToAsset', 'assetToEquity', 'CAToAsset', 'tangibleAssetToAsset', 'ebitTofloaterest', 'CFOToOR', 'CFOToNP', 'CFOToGr', 'dupontROE', 'dupontAssetStoEquity', 'dupontAssetTurn', 'dupontPnitoni', 'dupontNitogr', 'dupontTaxBurden', 'dupontfloatburden', 'dupontEbittogr', 'label', 'label_1d', 'label_3d', 'label_5d', 'label_7d']
        # self.record_defaults = [['mydefault'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        # self.select_col = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176]

        self.col_name = ['date', 'code', 'industry', 'open_ratio', 'close_ratio', 'high_ratio', 'low_ratio', 'open_ratio_7d_avg', 'close_ratio_7d_avg', 'high_ratio_7d_avg', 'low_ratio_7d_avg', 'amount', 'peTTM', 'pcfNcfTTM', 'pbMRQ', 'isST', 'turn', 'rsv_5', 'k_value_5', 'd_value_5', 'j_value_5', 'k_value_trend_5', 'kd_value5', 'rsv_9', 'k_value_9', 'd_value_9', 'j_value_9', 'k_value_trend_9', 'kd_value9', 'rsv_19', 'k_value_19', 'd_value_19', 'j_value_19', 'k_value_trend_19', 'kd_value19', 'rsv_73', 'k_value_73', 'd_value_73', 'j_value_73', 'k_value_trend_73', 'kd_value73', 'macd_positive', 'macd_dif_ratio', 'macd_dif_2', 'macd_dea_2', 'macd_2', 'macd_positive_ratio_2', 'macd_dif_3', 'macd_dea_3', 'macd_3', 'macd_positive_ratio_3', 'macd_dif_5', 'macd_dea_5', 'macd_5', 'macd_positive_ratio_5', 'macd_dif_10', 'macd_dea_10', 'macd_10', 'macd_positive_ratio_10', 'macd_dif_20', 'macd_dea_20', 'macd_20', 'macd_positive_ratio_20', 'macd_dif_40', 'macd_dea_40', 'macd_40', 'macd_positive_ratio_40', 'macd_dif_dea', 'width_20', 'close_mb20_diff', 'cr_3d', 'cr_5d', 'cr_10d', 'cr_20d', 'cr_40d', 'rsi_3d', 'rsi_5d', 'rsi_10d', 'rsi_20d', 'rsi_40d', 'turn_3d_avg', 'turn_3davg_dif', 'turn_3dmax_dif', 'turn_3dmin_dif', 'close_3davg_dif', 'close_3dmax_dif', 'close_3dmin_dif', 'close_3d_dif', 'turn_5d_avg', 'turn_5davg_dif', 'turn_5dmax_dif', 'turn_5dmin_dif', 'turn_3_5d_avg', 'close_5davg_dif', 'close_5dmax_dif', 'close_5dmin_dif', 'close_5d_dif', 'close_3_5d_avg', 'turn_10d_avg', 'turn_10davg_dif', 'turn_10dmax_dif', 'turn_10dmin_dif', 'turn_5_10d_avg', 'close_10davg_dif', 'close_10dmax_dif', 'close_10dmin_dif', 'close_10d_dif', 'close_5_10d_avg', 'turn_20d_avg', 'turn_20davg_dif', 'turn_20dmax_dif', 'turn_20dmin_dif', 'turn_10_20d_avg', 'close_20davg_dif', 'close_20dmax_dif', 'close_20dmin_dif', 'close_20d_dif', 'close_10_20d_avg', 'turn_30d_avg', 'turn_30davg_dif', 'turn_30dmax_dif', 'turn_30dmin_dif', 'turn_20_30d_avg', 'close_30davg_dif', 'close_30dmax_dif', 'close_30dmin_dif', 'close_30d_dif', 'close_20_30d_avg', 'turn_60d_avg', 'turn_60davg_dif', 'turn_60dmax_dif', 'turn_60dmin_dif', 'turn_30_60d_avg', 'close_60davg_dif', 'close_60dmax_dif', 'close_60dmin_dif', 'close_60d_dif', 'close_30_60d_avg', 'turn_120d_avg', 'turn_120davg_dif', 'turn_120dmax_dif', 'turn_120dmin_dif', 'turn_60_120d_avg', 'close_120davg_dif', 'close_120dmax_dif', 'close_120dmin_dif', 'close_120d_dif', 'close_60_120d_avg', 'turn_240d_avg', 'turn_240davg_dif', 'turn_240dmax_dif', 'turn_240dmin_dif', 'turn_120_240d_avg', 'close_240davg_dif', 'close_240dmax_dif', 'close_240dmin_dif', 'close_240d_dif', 'close_120_240d_avg', 'industry_open', 'industry_close', 'industry_preclose', 'industry_high', 'industry_low', 'industry_turn', 'industry_amount', 'industry_pctChg', 'industry_peTTM', 'industry_pcfNcfTTM', 'industry_pbMRQ', 'rise_ratio', 'industry_open_ratio', 'industry_close_ratio', 'industry_high_ratio', 'industry_low_ratio', 'industry_open_ratio_7d_avg', 'industry_close_ratio_7d_avg', 'industry_high_ratio_7d_avg', 'industry_low_ratio_7d_avg', 'industry_rsv_5', 'industry_k_value_5', 'industry_d_value_5', 'industry_j_value_5', 'industry_k_value_trend_5', 'industry_kd_value5', 'industry_rsv_9', 'industry_k_value_9', 'industry_d_value_9', 'industry_j_value_9', 'industry_k_value_trend_9', 'industry_kd_value9', 'industry_rsv_19', 'industry_k_value_19', 'industry_d_value_19', 'industry_j_value_19', 'industry_k_value_trend_19', 'industry_kd_value19', 'industry_rsv_73', 'industry_k_value_73', 'industry_d_value_73', 'industry_j_value_73', 'industry_k_value_trend_73', 'industry_kd_value73', 'industry_macd_positive', 'industry_macd_dif_ratio', 'industry_macd_dif_2', 'industry_macd_dea_2', 'industry_macd_2', 'industry_macd_positive_ratio_2', 'industry_macd_dif_3', 'industry_macd_dea_3', 'industry_macd_3', 'industry_macd_positive_ratio_3', 'industry_macd_dif_5', 'industry_macd_dea_5', 'industry_macd_5', 'industry_macd_positive_ratio_5', 'industry_macd_dif_10', 'industry_macd_dea_10', 'industry_macd_10', 'industry_macd_positive_ratio_10', 'industry_macd_dif_20', 'industry_macd_dea_20', 'industry_macd_20', 'industry_macd_positive_ratio_20', 'industry_macd_dif_40', 'industry_macd_dea_40', 'industry_macd_40', 'industry_macd_positive_ratio_40', 'industry_macd_dif_dea', 'industry_width_20', 'industry_close_mb20_diff', 'industry_cr_3d', 'industry_cr_5d', 'industry_cr_10d', 'industry_cr_20d', 'industry_cr_40d', 'industry_rsi_3d', 'industry_rsi_5d', 'industry_rsi_10d', 'industry_rsi_20d', 'industry_rsi_40d', 'industry_turn_3d_avg', 'industry_turn_3davg_dif', 'industry_turn_3dmax_dif', 'industry_turn_3dmin_dif', 'industry_close_3davg_dif', 'industry_close_3dmax_dif', 'industry_close_3dmin_dif', 'industry_close_3d_dif', 'industry_turn_5d_avg', 'industry_turn_5davg_dif', 'industry_turn_5dmax_dif', 'industry_turn_5dmin_dif', 'industry_turn_3_5d_avg', 'industry_close_5davg_dif', 'industry_close_5dmax_dif', 'industry_close_5dmin_dif', 'industry_close_5d_dif', 'industry_close_3_5d_avg', 'industry_turn_10d_avg', 'industry_turn_10davg_dif', 'industry_turn_10dmax_dif', 'industry_turn_10dmin_dif', 'industry_turn_5_10d_avg', 'industry_close_10davg_dif', 'industry_close_10dmax_dif', 'industry_close_10dmin_dif', 'industry_close_10d_dif', 'industry_close_5_10d_avg', 'industry_turn_20d_avg', 'industry_turn_20davg_dif', 'industry_turn_20dmax_dif', 'industry_turn_20dmin_dif', 'industry_turn_10_20d_avg', 'industry_close_20davg_dif', 'industry_close_20dmax_dif', 'industry_close_20dmin_dif', 'industry_close_20d_dif', 'industry_close_10_20d_avg', 'industry_turn_30d_avg', 'industry_turn_30davg_dif', 'industry_turn_30dmax_dif', 'industry_turn_30dmin_dif', 'industry_turn_20_30d_avg', 'industry_close_30davg_dif', 'industry_close_30dmax_dif', 'industry_close_30dmin_dif', 'industry_close_30d_dif', 'industry_close_20_30d_avg', 'industry_turn_60d_avg', 'industry_turn_60davg_dif', 'industry_turn_60dmax_dif', 'industry_turn_60dmin_dif', 'industry_turn_30_60d_avg', 'industry_close_60davg_dif', 'industry_close_60dmax_dif', 'industry_close_60dmin_dif', 'industry_close_60d_dif', 'industry_close_30_60d_avg', 'industry_turn_120d_avg', 'industry_turn_120davg_dif', 'industry_turn_120dmax_dif', 'industry_turn_120dmin_dif', 'industry_turn_60_120d_avg', 'industry_close_120davg_dif', 'industry_close_120dmax_dif', 'industry_close_120dmin_dif', 'industry_close_120d_dif', 'industry_close_60_120d_avg', 'industry_turn_240d_avg', 'industry_turn_240davg_dif', 'industry_turn_240dmax_dif', 'industry_turn_240dmin_dif', 'industry_turn_120_240d_avg', 'industry_close_240davg_dif', 'industry_close_240dmax_dif', 'industry_close_240dmin_dif', 'industry_close_240d_dif', 'industry_close_120_240d_avg', 'label_7', 'label_7_weight', 'label_7_max', 'label_7_max_weight', 'label_15', 'label_15_weight', 'label_15_max', 'label_15_max_weight']

        self.fea_config = {'date': {'bucket': 10, 'type': 'float'}, 'code': {'bucket': 100010, 'type': 'string'}, 'industry': {'bucket': 110, 'type': 'bucketId'}, 'open_ratio': {'bucket': 210, 'type': 'float'}, 'close_ratio': {'bucket': 210, 'type': 'float'}, 'high_ratio': {'bucket': 210, 'type': 'float'}, 'low_ratio': {'bucket': 210, 'type': 'float'}, 'open_ratio_7d_avg': {'bucket': 210, 'type': 'float'}, 'close_ratio_7d_avg': {'bucket': 210, 'type': 'float'}, 'high_ratio_7d_avg': {'bucket': 210, 'type': 'float'}, 'low_ratio_7d_avg': {'bucket': 210, 'type': 'float'}, 'amount': {'bucket': 10010, 'type': 'float2bucket'}, 'peTTM': {'bucket': 1510, 'type': 'float2bucket'}, 'pcfNcfTTM': {'bucket': 2010, 'type': 'float2bucket'}, 'pbMRQ': {'bucket': 2010, 'type': 'float2bucket'}, 'isST': {'bucket': 13, 'type': 'bucketId'}, 'turn': {'bucket': 210, 'type': 'float2bucket'}, 'rsv_5': {'bucket': 210, 'type': 'float'}, 'k_value_5': {'bucket': 210, 'type': 'float'}, 'd_value_5': {'bucket': 210, 'type': 'float'}, 'j_value_5': {'bucket': 210, 'type': 'float'}, 'k_value_trend_5': {'bucket': 210, 'type': 'float'}, 'kd_value5': {'bucket': 13, 'type': 'bucketId'}, 'rsv_9': {'bucket': 210, 'type': 'float'}, 'k_value_9': {'bucket': 210, 'type': 'float'}, 'd_value_9': {'bucket': 210, 'type': 'float'}, 'j_value_9': {'bucket': 210, 'type': 'float'}, 'k_value_trend_9': {'bucket': 210, 'type': 'float'}, 'kd_value9': {'bucket': 210, 'type': 'bucketId'}, 'rsv_19': {'bucket': 210, 'type': 'float'}, 'k_value_19': {'bucket': 210, 'type': 'float'}, 'd_value_19': {'bucket': 210, 'type': 'float'}, 'j_value_19': {'bucket': 210, 'type': 'float'}, 'k_value_trend_19': {'bucket': 210, 'type': 'float'}, 'kd_value19': {'bucket': 210, 'type': 'bucketId'}, 'rsv_73': {'bucket': 210, 'type': 'float'}, 'k_value_73': {'bucket': 210, 'type': 'float'}, 'd_value_73': {'bucket': 210, 'type': 'float'}, 'j_value_73': {'bucket': 210, 'type': 'float'}, 'k_value_trend_73': {'bucket': 210, 'type': 'float'}, 'kd_value73': {'bucket': 210, 'type': 'bucketId'}, 'macd_positive': {'bucket': 12, 'type': 'bucketId'}, 'macd_dif_ratio': {'bucket': 10, 'type': 'float'}, 'macd_dif_2': {'bucket': 12, 'type': 'bucketId'}, 'macd_dea_2': {'bucket': 12, 'type': 'bucketId'}, 'macd_2': {'bucket': 12, 'type': 'bucketId'}, 'macd_positive_ratio_2': {'bucket': 10, 'type': 'float'}, 'macd_dif_3': {'bucket': 12, 'type': 'bucketId'}, 'macd_dea_3': {'bucket': 12, 'type': 'bucketId'}, 'macd_3': {'bucket': 12, 'type': 'bucketId'}, 'macd_positive_ratio_3': {'bucket': 10, 'type': 'float'}, 'macd_dif_5': {'bucket': 12, 'type': 'bucketId'}, 'macd_dea_5': {'bucket': 12, 'type': 'bucketId'}, 'macd_5': {'bucket': 12, 'type': 'bucketId'}, 'macd_positive_ratio_5': {'bucket': 10, 'type': 'float'}, 'macd_dif_10': {'bucket': 12, 'type': 'bucketId'}, 'macd_dea_10': {'bucket': 12, 'type': 'bucketId'}, 'macd_10': {'bucket': 12, 'type': 'bucketId'}, 'macd_positive_ratio_10': {'bucket': 10, 'type': 'float'}, 'macd_dif_20': {'bucket': 12, 'type': 'bucketId'}, 'macd_dea_20': {'bucket': 12, 'type': 'bucketId'}, 'macd_20': {'bucket': 12, 'type': 'bucketId'}, 'macd_positive_ratio_20': {'bucket': 10, 'type': 'float'}, 'macd_dif_40': {'bucket': 12, 'type': 'bucketId'}, 'macd_dea_40': {'bucket': 12, 'type': 'bucketId'}, 'macd_40': {'bucket': 12, 'type': 'bucketId'}, 'macd_positive_ratio_40': {'bucket': 10, 'type': 'float'}, 'macd_dif_dea': {'bucket': 13, 'type': 'bucketId'}, 'width_20': {'bucket': 10, 'type': 'float'}, 'close_mb20_diff': {'bucket': 10, 'type': 'float'}, 'cr_3d': {'bucket': 110, 'type': 'float2bucket'}, 'cr_5d': {'bucket': 110, 'type': 'float2bucket'}, 'cr_10d': {'bucket': 110, 'type': 'float2bucket'}, 'cr_20d': {'bucket': 110, 'type': 'float2bucket'}, 'cr_40d': {'bucket': 110, 'type': 'float2bucket'}, 'rsi_3d': {'bucket': 10, 'type': 'float'}, 'rsi_5d': {'bucket': 10, 'type': 'float'}, 'rsi_10d': {'bucket': 10, 'type': 'float'}, 'rsi_20d': {'bucket': 10, 'type': 'float'}, 'rsi_40d': {'bucket': 10, 'type': 'float'}, 'turn_3d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_3davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_3dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_3dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_3davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_3dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'close_3dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_3d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_5d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_5davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_5dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_5dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_3_5d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'close_5davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_5dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'close_5dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_5d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_3_5d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_10d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_10davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_10dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_10dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_5_10d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'close_10davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_10dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'close_10dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_10d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_5_10d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_20d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_20davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_20dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_20dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_10_20d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'close_20davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_20dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'close_20dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_20d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_10_20d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_30d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_30davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_30dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_30dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_20_30d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'close_30davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_30dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'close_30dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_30d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_20_30d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_60d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_60davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_60dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_60dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_30_60d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'close_60davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_60dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'close_60dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_60d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_30_60d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_120d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_120davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_120dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_120dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_60_120d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'close_120davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_120dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'close_120dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_120d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_60_120d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_240d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'turn_240davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_240dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_240dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'turn_120_240d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'close_240davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_240dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'close_240dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_240d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'close_120_240d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_open': {'bucket': 210, 'type': 'float2bucket'}, 'industry_close': {'bucket': 210, 'type': 'float2bucket'}, 'industry_preclose': {'bucket': 210, 'type': 'float2bucket'}, 'industry_high': {'bucket': 210, 'type': 'float2bucket'}, 'industry_low': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn': {'bucket': 210, 'type': 'float2bucket'}, 'industry_amount': {'bucket': 10010, 'type': 'float2bucket'}, 'industry_pctChg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_peTTM': {'bucket': 1510, 'type': 'float2bucket'}, 'industry_pcfNcfTTM': {'bucket': 2010, 'type': 'float2bucket'}, 'industry_pbMRQ': {'bucket': 2010, 'type': 'float2bucket'}, 'rise_ratio': {'bucket': 10, 'type': 'float'}, 'industry_open_ratio': {'bucket': 10, 'type': 'float'}, 'industry_close_ratio': {'bucket': 10, 'type': 'float'}, 'industry_high_ratio': {'bucket': 10, 'type': 'float'}, 'industry_low_ratio': {'bucket': 10, 'type': 'float'}, 'industry_open_ratio_7d_avg': {'bucket': 10, 'type': 'float'}, 'industry_close_ratio_7d_avg': {'bucket': 10, 'type': 'float'}, 'industry_high_ratio_7d_avg': {'bucket': 10, 'type': 'float'}, 'industry_low_ratio_7d_avg': {'bucket': 10, 'type': 'float'}, 'industry_rsv_5': {'bucket': 10, 'type': 'float'}, 'industry_k_value_5': {'bucket': 10, 'type': 'float'}, 'industry_d_value_5': {'bucket': 10, 'type': 'float'}, 'industry_j_value_5': {'bucket': 10, 'type': 'float'}, 'industry_k_value_trend_5': {'bucket': 10, 'type': 'float'}, 'industry_kd_value5': {'bucket': 13, 'type': 'bucketId'}, 'industry_rsv_9': {'bucket': 10, 'type': 'float'}, 'industry_k_value_9': {'bucket': 10, 'type': 'float'}, 'industry_d_value_9': {'bucket': 10, 'type': 'float'}, 'industry_j_value_9': {'bucket': 10, 'type': 'float'}, 'industry_k_value_trend_9': {'bucket': 10, 'type': 'float'}, 'industry_kd_value9': {'bucket': 13, 'type': 'bucketId'}, 'industry_rsv_19': {'bucket': 10, 'type': 'float'}, 'industry_k_value_19': {'bucket': 10, 'type': 'float'}, 'industry_d_value_19': {'bucket': 10, 'type': 'float'}, 'industry_j_value_19': {'bucket': 10, 'type': 'float'}, 'industry_k_value_trend_19': {'bucket': 10, 'type': 'float'}, 'industry_kd_value19': {'bucket': 13, 'type': 'bucketId'}, 'industry_rsv_73': {'bucket': 10, 'type': 'float'}, 'industry_k_value_73': {'bucket': 10, 'type': 'float'}, 'industry_d_value_73': {'bucket': 10, 'type': 'float'}, 'industry_j_value_73': {'bucket': 10, 'type': 'float'}, 'industry_k_value_trend_73': {'bucket': 10, 'type': 'float'}, 'industry_kd_value73': {'bucket': 13, 'type': 'bucketId'}, 'industry_macd_positive': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_dif_ratio': {'bucket': 210, 'type': 'float'}, 'industry_macd_dif_2': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_dea_2': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_2': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_positive_ratio_2': {'bucket': 10, 'type': 'float'}, 'industry_macd_dif_3': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_dea_3': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_3': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_positive_ratio_3': {'bucket': 10, 'type': 'float'}, 'industry_macd_dif_5': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_dea_5': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_5': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_positive_ratio_5': {'bucket': 10, 'type': 'float'}, 'industry_macd_dif_10': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_dea_10': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_10': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_positive_ratio_10': {'bucket': 10, 'type': 'float'}, 'industry_macd_dif_20': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_dea_20': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_20': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_positive_ratio_20': {'bucket': 10, 'type': 'float'}, 'industry_macd_dif_40': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_dea_40': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_40': {'bucket': 12, 'type': 'bucketId'}, 'industry_macd_positive_ratio_40': {'bucket': 10, 'type': 'float'}, 'industry_macd_dif_dea': {'bucket': 13, 'type': 'bucketId'}, 'industry_width_20': {'bucket': 10, 'type': 'float'}, 'industry_close_mb20_diff': {'bucket': 10, 'type': 'float'}, 'industry_cr_3d': {'bucket': 110, 'type': 'float2bucket'}, 'industry_cr_5d': {'bucket': 110, 'type': 'float2bucket'}, 'industry_cr_10d': {'bucket': 110, 'type': 'float2bucket'}, 'industry_cr_20d': {'bucket': 110, 'type': 'float2bucket'}, 'industry_cr_40d': {'bucket': 110, 'type': 'float2bucket'}, 'industry_rsi_3d': {'bucket': 10, 'type': 'float'}, 'industry_rsi_5d': {'bucket': 10, 'type': 'float'}, 'industry_rsi_10d': {'bucket': 10, 'type': 'float'}, 'industry_rsi_20d': {'bucket': 10, 'type': 'float'}, 'industry_rsi_40d': {'bucket': 10, 'type': 'float'}, 'industry_turn_3d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_3davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_3dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_3dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_3davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_3dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'industry_close_3dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_3d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_5d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_5davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_5dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_5dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_3_5d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_close_5davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_5dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'industry_close_5dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_5d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_3_5d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_10d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_10davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_10dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_10dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_5_10d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_close_10davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_10dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'industry_close_10dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_10d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_5_10d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_20d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_20davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_20dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_20dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_10_20d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_close_20davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_20dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'industry_close_20dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_20d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_10_20d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_30d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_30davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_30dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_30dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_20_30d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_close_30davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_30dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'industry_close_30dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_30d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_20_30d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_60d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_60davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_60dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_60dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_30_60d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_close_60davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_60dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'industry_close_60dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_60d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_30_60d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_120d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_120davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_120dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_120dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_60_120d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_close_120davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_120dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'industry_close_120dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_120d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_60_120d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_240d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_turn_240davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_240dmax_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_240dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_turn_120_240d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'industry_close_240davg_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_240dmax_dif': {'bucket': 110, 'type': 'float2bucket'}, 'industry_close_240dmin_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_240d_dif': {'bucket': 510, 'type': 'float2bucket'}, 'industry_close_120_240d_avg': {'bucket': 210, 'type': 'float2bucket'}, 'label_7': {'bucket': 10, 'type': 'float'}, 'label_7_weight': {'bucket': 10, 'type': 'float'}, 'label_7_max': {'bucket': 10, 'type': 'float'}, 'label_7_max_weight': {'bucket': 10, 'type': 'float'}, 'label_15': {'bucket': 10, 'type': 'float'}, 'label_15_weight': {'bucket': 10, 'type': 'float'}, 'label_15_max': {'bucket': 10, 'type': 'float'}, 'label_15_max_weight': {'bucket': 10, 'type': 'float'}}
        self.record_defaults = [[0.0], ['mydefault'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        self.select_col = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        self.get_fea_columns()
        self.dnn_dims = self.init_variable()
        self.fea_count = sum(self.select_col) - self.label_cnt

    def init_variable(self):
        dnn_hidden_1 = 1024
        dnn_hidden_2 = 512
        dnn_hidden_3 = 256
        dnn_hidden_4 = 128
        dnn_hidden_5 = 64
        # dnn_hidden_6 = 32
        # dnn_dims = [dnn_hidden_1, dnn_hidden_2, dnn_hidden_3, dnn_hidden_4, dnn_hidden_5, dnn_hidden_6]
        dnn_dims = [dnn_hidden_1, dnn_hidden_2, dnn_hidden_3, dnn_hidden_4, dnn_hidden_5]
        return dnn_dims

    def decode_csv(self, line):
        data = tf.decode_csv(line, record_defaults=self.record_defaults, field_delim=',', use_quote_delim=True, na_value='', name=None)
        label = data[-self.label_cnt:]
        features = {}
        for index in range(len(self.col_name) - self.label_cnt):
            if self.select_col[index] == 1:
                fea_name = self.col_name[index]
                config = self.fea_config[fea_name]
                fea_type = config['type']
                if fea_type in ('mydefault'):
                    val = tf.cast(data[index], dtype=tf.string)
                elif fea_type in ('bucketId', 'float2bucket'):
                    val = tf.cast(data[index], dtype=tf.int32)
                elif fea_type in ('float'):
                    val = tf.cast(data[index], dtype=tf.float32)
                key = self.col_name[index]
                features[key] = val

        features['label_7'] = label[0]
        features['label_7_weight'] = label[1]
        features['label_7_max'] = label[2]
        features['label_7_max_weight'] = label[3]
        features['label_15'] = label[4]
        features['label_15_weight'] = label[5]
        features['label_15_max'] = label[6]
        features['label_15_max_weight'] = label[7]

        features['date'] = data[0]
        features['code'] = tf.cast(data[1], dtype=tf.string)
        return features, label

    def train_input_fn_from_csv(self, data_path, epoch=1, batch_size=1024):
        with tf.device('/cpu:0'):
            dataset = tf.data.TextLineDataset(data_path).skip(1)
            dataset = dataset.repeat(epoch).shuffle(buffer_size=batch_size*10).batch(batch_size)
            dataset = dataset.map(self.decode_csv, num_parallel_calls=20).prefetch(batch_size*10)
            dataset = dataset.make_one_shot_iterator()
            features_batch, label_batch = dataset.get_next()
            print('=================iterator============')
        return features_batch, label_batch

    def test_input_fn_from_csv(self, data_path, epoch=1, batch_size=1024):
        with tf.device('/cpu:0'):
            dataset = tf.data.TextLineDataset(data_path).skip(1)
            dataset = dataset.batch(batch_size).repeat(epoch)
            dataset = dataset.map(self.decode_csv, num_parallel_calls=10).prefetch(1000)
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
    # def tag_embedding(self, key_name, hash_bucket, dim):
    #     id_feature = layers.sparse_column_with_hash_bucket(
    #         column_name=key_name,
    #         hash_bucket_size=hash_bucket,
    #         combiner='mean',
    #         dtype=tf.string,
    #     )
    #     emb_col = layers.embedding_column(
    #         id_feature,
    #         dimension=dim,
    #         combiner='mean',initializer=tf.truncated_normal_initializer(stddev=self.stddev)
    #     )
    #     return emb_col

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

    def get_fea_columns(self):
        for field_idx in range(len(self.col_name) - self.label_cnt):
            if self.select_col[field_idx] == 1:
                fea_name = self.col_name[field_idx]
                config = self.fea_config[fea_name]
                fea_bucket = config['bucket']
                fea_type = config['type']
                # fea_field = config['field']

                if fea_type in ('mydefault'):
                    value = self.hash_embedding(fea_name, fea_bucket, self.embedding_dim)
                elif fea_type in ('bucketId', 'float2bucket'):
                    value = self.index_embedding(fea_name, fea_bucket, self.embedding_dim)
                elif fea_type in ('float'):
                    value = self.numeric_column(fea_name)
                self.feature_columns_dict['other_feas'].append(value)

    def model_fn_params(self):
        model_fn_params_dict = {'lr': self.lr, 'optimizer': self.optimizer, 'feature_columns': self.feature_columns_dict}
        return model_fn_params_dict

    def dnn(self, net):
        for idx, units in enumerate(self.dnn_dims):
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu, name='dnn_' + str(idx))
            net = tf.layers.batch_normalization(inputs=net, name='concat_bn' + str(idx), reuse=tf.AUTO_REUSE)
        net = tf.layers.dense(net, units=1)
        return net

    def model_fn(self, features, labels, mode, params):
        #
        other_feas = params['feature_columns']['other_feas']

        with tf.device('/cpu:0'):
            all_feas = []
            all_feas = all_feas + other_feas
            embed_input = tf.feature_column.input_layer(features, all_feas)
            # embed_input = tf.reshape(embed_input, [-1, self.embedding_dim * self.fea_count])
        with tf.device('/gpu:0'):
            y_ctr = self.dnn(embed_input)
        # logging.warning('y_ctr.device: {}'.format(y_ctr.device))
        #
        # if mode == tf.estimator.ModeKeys.TRAIN and self.checkpoint_path != self.output_dir:
        #     self.init_variables_from_checkpoint()
        ctr_pred = tf.sigmoid(y_ctr, name="prediction")
        ctr_pred = tf.reshape(ctr_pred, [-1])
        # ctr_pred = tf.reshape(y_ctr, [-1])
        # predict
        # label_1d, label_3d, label_5d, label_7d
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'predition': ctr_pred,
                'label_7': features['label_7'],
                'label_7_weight': features['label_7_weight'],
                'label_7_max': features['label_7_max'],
                'label_7_max_weight': features['label_7_max_weight'],
                'label_15': features['label_15'],
                'label_15_weight': features['label_15_weight'],
                'label_15_max': features['label_15_max'],
                'label_15_max_weight': features['label_15_max_weight'],
                'code': features['code'],
                'date': features['date']
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        # ground_truth_ctr = (labels[0, :] + labels[1, :] + labels[2, :] + labels[3, :] + labels[4, :])/5

        # features['label_7'] = label[0]
        # features['label_7_weight'] = label[1]
        # features['label_7_max'] = label[2]
        # features['label_7_max_weight'] = label[3]
        # features['label_15'] = label[4]
        # features['label_15_weight'] = label[5]
        # features['label_15_max'] = label[6]
        # features['label_15_max_weight'] = label[7]

        ground_truth_ctr = tf.reshape(labels[0, :], [-1])
        ground_truth_ctr_weight = tf.reshape(labels[1, :], [-1])

        # loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=ground_truth_ctr, predictions=ctr_pred, weights=ground_truth_ctr_weight))
        loss = tf.reduce_mean(tf.losses.log_loss(labels=ground_truth_ctr, predictions=ctr_pred, weights=ground_truth_ctr_weight))

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
                input_fn=lambda: self.train_input_fn_from_csv(data_path=self.train_data, epoch=1, batch_size=self.batch_size),
                max_steps=self.max_train_step)
            eval_spec = tf.estimator.EvalSpec(
                input_fn=lambda: self.train_input_fn_from_csv(data_path=self.test_data, epoch=1, batch_size=20000),throttle_secs=300)

            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

            # if self.is_chief:
            #     print("......................Start savemodel......................")
            #     classifier.export_savedmodel(self.output_dir, self.serving_input_receiver_fn, strip_default_attrs=True)
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
                                           'predition': list(predict_result['predition']),
                                           'label_7d': list(predict_result['label_7d']),
                                           'label_7_weight': list(predict_result['label_7_weight']),
                                           'label_7_max': list(predict_result['label_7_max']),
                                           'label_7_max_weight': list(predict_result['label_7_max_weight']),
                                           'label_15': list(predict_result['label_15']),
                                           'label_15_weight': list(predict_result['label_15_weight']),
                                           'label_15_max': list(predict_result['label_15_max']),
                                           'label_15_max_weight': list(predict_result['label_15_max_weight'])
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
    model = DeepFM(2009)
    features_batch, label_batch = model.train_input_fn_from_csv(data_path=model.test_data, epoch=1, batch_size=3)
    # path = 'E:/pythonProject/future/data/datafile/test1'
    path = 'E:/pythonProject/future/data/datafile/test2.csv'
    # features_batch, label_batch = model.train_input_fn_from_csv_test(data_path=model.test_data, epoch=1, batch_size=3)
    # dataset = model.train_input_fn_from_csv_test(data_path=path, epoch=1, batch_size=3)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        # print(sess.run([data]))
        # print(sess.run([label_batch]))
        # print(sess.run([features_batch, label_batch]))
        # print(sess.run(dataset))
        print(sess.run([features_batch, label_batch]))
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

    # 2008, 2009, 2010,2011,
    years = [2014, 2015, 2016, 2017, 2018, 2019]
    for year in years:
        model = DeepFM(year, 'train', 'model_v4')
        model.run()

        # model = DeepFM(year, 'predict', 'model_v3')
        # model.run()
    # model = DeepFM(2013, 'train', 'model_v4')
    # model.run()