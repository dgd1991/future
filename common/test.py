# -*- coding: utf-8 -*-
"""Summary
"""
import os
import re
import sys
import pandas as pd
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
    evaluate_result = 'E:/pythonProject/future/data/datafile/prediction_result/saved_model_v64/evaluate_result_2022.csv'
    result = pd.read_csv(evaluate_result)
    result_map = {}
    for tup in result.itertuples():
        if tup[2]< 0.6006281 or tup[3]>0.30227876:
            result_map[tup[1]] = ' 1 '
        else:
            result_map[tup[1]] = ' 0 '
    kv_map = {}
    lineNum = -3
    Note = open(os.path.join(commonPath, modelName + '_new'), mode='a+')

    with open(conf_file, "r") as fin:
        for _ in range(3):
            line = fin.readline()
            lineNum += 1
            print(lineNum)

        index = 0
        while line:
            # 从配置文件中读取特征的 featureName, dataOperation,fieldLength,origin,feedName,logNum | floor | ceil | bucketNum | base
            lineNum += 1
            print(lineNum)
            name, type, default_value, is_used, bucket, dim = re.split("\|", line.strip())
            if name.strip() in result_map:
                is_used = result_map[name.strip()]
            new_line = name + "|" + type + "|" + default_value + "|" + is_used + "|" + bucket + "|" + dim + '\n'
            Note.write(new_line)
            index += 1
            line = fin.readline()
if __name__ == '__main__':
    confPath = "E:/pythonProject/future/common/"
    modelName = "model_v66"
    parse_feaconf(confPath, modelName)
