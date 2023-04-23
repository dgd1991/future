# -*- coding: utf-8 -*-
"""Summary
"""
import os
import re
import sys
import argparse

sys.path.append("../common")

def parse_feaconf(commonPath, modelName):
    """Summary

    Args:
        conf_file (TYPE): Description

    Returns:
        TYPE: Description
    """
    conf_file = os.path.join(commonPath, modelName)
    default_value_list = []
    col_name = []
    select_col = []
    all_fea_config = {}
    feature_accuracy = {}
    # del1
    # fea_need_delete = ['code_market', 'market_value', 'macd_dif_2', 'macd_2', 'macd_dif_3', 'macd_3', 'macd_dif_5', 'macd_5', 'macd_dif_10', 'macd_dea_10', 'macd_10', 'macd_dif_20', 'macd_dea_20', 'macd_20', 'macd_dea_40', 'macd_40', 'turn_3davg_dif', 'turn_5davg_dif', 'turn_3_5d_avg', 'turn_3_5dmax_dif', 'turn_3_5dmin_dif', 'close_3_5d_avg', 'turn_10davg_dif', 'turn_5_10d_avg', 'turn_5_10dmax_dif', 'turn_5_10dmin_dif', 'close_5_10d_avg', 'turn_20davg_dif', 'turn_10_20d_avg', 'turn_10_20dmax_dif', 'turn_10_20dmin_dif', 'turn_20_30d_avg', 'turn_20_30dmax_dif', 'turn_20_30dmin_dif', 'turn_30_60d_avg', 'turn_30_60dmax_dif', 'turn_30_60dmin_dif', 'turn_60_120d_avg', 'turn_60_120dmax_dif', 'turn_60_120dmin_dif', 'turn_120_240dmax_dif', 'max_turn_index3d', 'max_close_index3d', 'min_close_index3d', 'max_turn_close3d', 'max_closs_turn3d', 'min_closs_turn3d', 'max_turn_index5d', 'max_turn_close5d', 'max_closs_turn5d', 'min_closs_turn5d', 'max_turn_index10d', 'max_closs_turn10d', 'min_closs_turn10d', 'max_turn_index20d', 'max_closs_turn20d', 'max_turn_index30d', 'max_closs_turn30d', 'max_turn_index60d', 'max_turn_index120d']
    # del2，效果负向
    # fea_need_delete = ['open_ratio_7d_avg','pctChg_greater_13_5d','pctChg_less_13_10d']
    # del3
    # fea_need_delete = ['isST', 'macd_2', 'macd_3', 'macd_5', 'macd_dif_dea', 'turn_3davg_dif', 'turn_3_5dmax_dif', 'turn_3_5dmin_dif', 'turn_5_10dmax_dif', 'turn_5_10dmin_dif', 'max_turn_index3d', 'max_turn_index5d']
    # industry1_del1，auc效果持平，loss负向
    # fea_need_delete = ['industry_id_level1_open_ratio', 'industry_id_level1_open_ratio_7d_avg', 'industry_id_level1_macd_positive_ratio_20']
    # industry1_del2
    fea_need_delete = ['industry_id_level3_turn', 'industry_id_level3_peTTM', 'industry_id_level3_pcfNcfTTM', 'industry_id_level3_pbMRQ', 'industry_id_level3_market_value', 'industry_id_level3_pctChg_up_limit', 'industry_id_level3_pctChg_down_limit', 'industry_id_level3_pctChg_up_limit_3', 'industry_id_level3_pctChg_down_limit_3', 'industry_id_level3_pctChg_up_limit_7', 'industry_id_level3_pctChg_down_limit_7', 'industry_id_level3_pctChg_up_limit_15', 'industry_id_level3_pctChg_down_limit_15', 'industry_id_level3_pctChg_up_limit_30', 'industry_id_level3_pctChg_down_limit_30', 'industry_id_level3_macd_positive', 'industry_id_level3_macd_dif_2', 'industry_id_level3_macd_dea_2', 'industry_id_level3_macd_2', 'industry_id_level3_macd_dif_3', 'industry_id_level3_macd_dea_3', 'industry_id_level3_macd_3', 'industry_id_level3_macd_dif_5', 'industry_id_level3_macd_dea_5', 'industry_id_level3_macd_5', 'industry_id_level3_macd_dif_10', 'industry_id_level3_macd_dea_10', 'industry_id_level3_macd_10', 'industry_id_level3_macd_dif_20', 'industry_id_level3_macd_dea_20', 'industry_id_level3_macd_20', 'industry_id_level3_macd_dif_40', 'industry_id_level3_macd_dea_40', 'industry_id_level3_macd_40', 'industry_id_level3_macd_dif_dea', 'industry_id_level3_cr_26d', 'industry_id_level3_cr_trend_26d', 'industry_id_level3_cr_trend_26d_0', 'industry_id_level3_cr_trend_26d_1', 'industry_id_level3_cr_trend_26d_2', 'industry_id_level3_cr_trend_26d_3', 'industry_id_level3_turn_3davg_dif', 'industry_id_level3_close_3davg_dif', 'industry_id_level3_close_3dmax_dif', 'industry_id_level3_close_3dmin_dif', 'industry_id_level3_close_3d_dif', 'industry_id_level3_turn_5davg_dif', 'industry_id_level3_turn_3_5d_avg', 'industry_id_level3_turn_3_5dmax_dif', 'industry_id_level3_turn_3_5dmin_dif', 'industry_id_level3_close_5davg_dif', 'industry_id_level3_close_5dmax_dif', 'industry_id_level3_close_5dmin_dif', 'industry_id_level3_close_5d_dif', 'industry_id_level3_close_3_5d_avg', 'industry_id_level3_turn_10davg_dif', 'industry_id_level3_turn_5_10d_avg', 'industry_id_level3_turn_5_10dmax_dif', 'industry_id_level3_turn_5_10dmin_dif', 'industry_id_level3_close_10davg_dif', 'industry_id_level3_close_10dmax_dif', 'industry_id_level3_close_10dmin_dif', 'industry_id_level3_close_10d_dif', 'industry_id_level3_close_5_10d_avg', 'industry_id_level3_turn_20davg_dif', 'industry_id_level3_turn_10_20d_avg', 'industry_id_level3_turn_10_20dmax_dif', 'industry_id_level3_turn_10_20dmin_dif', 'industry_id_level3_close_20davg_dif', 'industry_id_level3_close_20dmax_dif', 'industry_id_level3_close_20dmin_dif', 'industry_id_level3_close_20d_dif', 'industry_id_level3_close_10_20d_avg', 'industry_id_level3_turn_30davg_dif', 'industry_id_level3_turn_20_30d_avg', 'industry_id_level3_turn_20_30dmax_dif', 'industry_id_level3_turn_20_30dmin_dif', 'industry_id_level3_close_30davg_dif', 'industry_id_level3_close_30dmax_dif', 'industry_id_level3_close_30dmin_dif', 'industry_id_level3_close_30d_dif', 'industry_id_level3_close_20_30d_avg', 'industry_id_level3_turn_60davg_dif', 'industry_id_level3_turn_30_60d_avg', 'industry_id_level3_turn_30_60dmax_dif', 'industry_id_level3_turn_30_60dmin_dif', 'industry_id_level3_close_60davg_dif', 'industry_id_level3_close_60dmax_dif', 'industry_id_level3_close_60dmin_dif', 'industry_id_level3_close_60d_dif', 'industry_id_level3_close_30_60d_avg', 'industry_id_level3_turn_120davg_dif', 'industry_id_level3_turn_60_120d_avg', 'industry_id_level3_turn_60_120dmax_dif', 'industry_id_level3_turn_60_120dmin_dif', 'industry_id_level3_close_120davg_dif', 'industry_id_level3_close_120dmax_dif', 'industry_id_level3_close_120dmin_dif', 'industry_id_level3_close_120d_dif', 'industry_id_level3_close_60_120d_avg', 'industry_id_level3_turn_240davg_dif', 'industry_id_level3_turn_120_240d_avg', 'industry_id_level3_turn_120_240dmax_dif', 'industry_id_level3_turn_120_240dmin_dif', 'industry_id_level3_close_240davg_dif', 'industry_id_level3_close_240dmax_dif', 'industry_id_level3_close_240dmin_dif', 'industry_id_level3_close_240d_dif', 'industry_id_level3_close_120_240d_avg', 'industry_id_level3_max_turn_index3d', 'industry_id_level3_max_close_index3d', 'industry_id_level3_min_close_index3d', 'industry_id_level3_max_turn_close3d', 'industry_id_level3_max_closs_turn3d', 'industry_id_level3_min_closs_turn3d', 'industry_id_level3_max_turn_index5d', 'industry_id_level3_max_close_index5d', 'industry_id_level3_min_close_index5d', 'industry_id_level3_max_turn_close5d', 'industry_id_level3_max_closs_turn5d', 'industry_id_level3_min_closs_turn5d', 'industry_id_level3_max_turn_index10d', 'industry_id_level3_max_close_index10d', 'industry_id_level3_min_close_index10d', 'industry_id_level3_max_turn_close10d', 'industry_id_level3_max_closs_turn10d', 'industry_id_level3_min_closs_turn10d', 'industry_id_level3_max_turn_index20d', 'industry_id_level3_max_close_index20d', 'industry_id_level3_min_close_index20d', 'industry_id_level3_max_turn_close20d', 'industry_id_level3_max_closs_turn20d', 'industry_id_level3_min_closs_turn20d', 'industry_id_level3_max_turn_index30d', 'industry_id_level3_max_close_index30d', 'industry_id_level3_min_close_index30d', 'industry_id_level3_max_turn_close30d', 'industry_id_level3_max_closs_turn30d', 'industry_id_level3_min_closs_turn30d', 'industry_id_level3_max_turn_index60d', 'industry_id_level3_max_close_index60d', 'industry_id_level3_min_close_index60d', 'industry_id_level3_max_turn_close60d', 'industry_id_level3_max_closs_turn60d', 'industry_id_level3_min_closs_turn60d', 'industry_id_level3_max_turn_index120d', 'industry_id_level3_max_close_index120d', 'industry_id_level3_min_close_index120d', 'industry_id_level3_max_turn_close120d', 'industry_id_level3_max_closs_turn120d', 'industry_id_level3_min_closs_turn120d', 'industry_id_level3_max_turn_index240d', 'industry_id_level3_max_close_index240d', 'industry_id_level3_min_close_index240d', 'industry_id_level3_max_turn_close240d', 'industry_id_level3_max_closs_turn240d', 'industry_id_level3_min_closs_turn240d']

    kv_map = {}
    lineNum = -3
    if os.path.isfile(os.path.join(confPath, modelName + "_deleted")):
        os.remove(os.path.join(confPath, modelName + "_deleted"))
    new_config_file = open(os.path.join(confPath, modelName + "_deleted"), "a")
    new_config_file.write("name | type | default_value | is_used | bucket | dim | shared_field")
    new_config_file.write("\n")
    new_config_file.write("-----|------|---------------|--------|--------|--------|----------")
    # new_config_file.write("\n")

    with open(conf_file, "r") as fin:
        for _ in range(3):
            line = fin.readline()
            lineNum += 1
            print(lineNum)

        index = 0
        while line:
            # 从配置文件中读取特征的 featureName, dataOperation,fieldLength,origin,feedName,logNum | floor | ceil | bucketNum | base
            lineNum += 1
            fea_config = {}
            print(lineNum)
            name, type, default_value, is_used, bucket, dim, shared_field = re.split("\|", line.strip())
            if is_used.strip() == '1' and fea_need_delete.__contains__(name.strip()):
                line = fin.readline()
            else:
                new_line = name + "|" + type + "|" + default_value + "|" + is_used + "|" + bucket + "|" + dim + "|" + shared_field
                new_config_file.write("\n")
                new_config_file.write(new_line)
                line = fin.readline()
    return col_name, select_col, default_value_list, all_fea_config, feature_accuracy
if __name__ == '__main__':
    confPath = "E:/pythonProject/future/common/"
    modelName = "model_v15"
    col_name, select_col, default_value_list, all_fea_config, feature_accuracy = parse_feaconf(confPath, modelName)