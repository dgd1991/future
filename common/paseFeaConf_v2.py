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

    kv_map = {}
    lineNum = -3
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
            name, type, default_value, is_used, bucket, dim = re.split("\|", line.strip())
            type = type.strip()
            is_used = int(is_used.strip())
            bucket = int(bucket.strip()) + 10
            fea_config['bucket'] = bucket
            fea_config['dim'] = int(dim.strip())
            fea_config['type'] = type.strip()
            if type == 'float':
                feature_accuracy[name] = 5

            if type == 'string':
                default_value_list.append([default_value.strip()])
            elif type in ('int', 'float2bucket', 'bucketId'):
                default_value_list.append([float(default_value.strip())])
            elif type == 'float':
                default_value_list.append([float(default_value.strip())])
            elif type == 'bigint':
                default_value_list.append([int(default_value.strip())])
            name = name.strip()
            assert (len(name) > 0)
            col_name.append(name)
            select_col.append(is_used)
            index += 1
            all_fea_config[name] = fea_config
            line = fin.readline()
    return col_name, select_col, default_value_list, all_fea_config, feature_accuracy
if __name__ == '__main__':
    confPath = "E:/pythonProject/future/common/"
    modelName = "model_v62"
    col_name, select_col, default_value_list, all_fea_config, feature_accuracy = parse_feaconf(confPath, modelName)
    with open(os.path.join(confPath, modelName + "_parsed"), "w") as fout:
        fout.write('self.col_name = ' + str(col_name))
        fout.write("\n")
        fout.write('self.select_col = ' + str(select_col))
        fout.write("\n")
        fout.write('self.record_defaults = ' + str(default_value_list))
        fout.write("\n")
        fout.write('self.fea_config = ' + str(all_fea_config))
        fout.write("\n")
        fout.write(str(feature_accuracy))