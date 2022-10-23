import tensorflow as tf
import logging
import os
import json
from datetime import datetime
# from tensorflow.contrib import layers
# from tensorflow import feature_column
from tensorflow.python.training import queue_runner_impl
from tensorflow.core.util.event_pb2 import SessionLog
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class DeepFM(object):
    def __init__(self):
        self.k_file_name = 'E:/pythonProject/future/data/datafile/code_k_data.csv'
        self.industry_file_name = 'E:/pythonProject/future/data/datafile/industry_data.csv'
        self.quarter_file_name = 'E:/pythonProject/future/data/datafile/quarter_data.csv'
        self.feature_file = 'E:/pythonProject/future/data/datafile/feature_all.csv'
        self.label_file = 'E:/pythonProject/future/data/datafile/label.csv'
        self.train_data = 'E:/pythonProject/future/data/datafile/train_data_category.csv'
        self.test_data = 'E:/pythonProject/future/data/datafile/test_data_category.csv'
        self.task_type = 'train'
        # self.checkpoint_path = "E:/pythonProject/future/saved_model"
        self.checkpoint_path = "E:\\pythonProject\\future\\saved_model"
        self.save_summary_steps = 1000
        self.save_checkpoint_and_eval_step = 1000
        self.every_n_steps = 1000
        self.max_train_step = 1000000
        self.embedding_dim = 10
        self.batch_size = 64
        self.feature_columns_dict = {'other_feas': []}
        self.lr = 0.001
        self.optimizer = 'Adam'
        self.stddev = 0.1
        # self.col_name = ['code', 'open_ratio', 'high_ratio', 'low_ratio', 'turn', 'preclose', 'amount', 'pctChg', 'peTTM', 'pcfNcfTTM', 'pbMRQ', 'isST', 'open_ratio_1d', 'high_ratio_1d', 'low_ratio_1d', 'turn_1d', 'pctChg_1d', 'open_ratio_2d', 'high_ratio_2d', 'low_ratio_2d', 'turn_2d', 'pctChg_2d', 'open_ratio_3d', 'high_ratio_3d', 'low_ratio_3d', 'turn_3d', 'pctChg_3d', 'open_ratio_4d', 'high_ratio_4d', 'low_ratio_4d', 'turn_4d', 'pctChg_4d', 'open_ratio_7d', 'high_ratio_7d', 'low_ratio_7d', 'turn_7d', 'pctChg_7d', 'open_ratio_10d', 'high_ratio_10d', 'low_ratio_10d', 'turn_10d', 'pctChg_10d', 'open_ratio_13d', 'high_ratio_13d', 'low_ratio_13d', 'turn_13d', 'pctChg_13d', 'open_ratio_16d', 'high_ratio_16d', 'low_ratio_16d', 'turn_16d', 'pctChg_16d', 'open_ratio_19d', 'high_ratio_19d', 'low_ratio_19d', 'turn_19d', 'pctChg_19d', 'open_ratio_22d', 'high_ratio_22d', 'low_ratio_22d', 'turn_22d', 'pctChg_22d', 'open_ratio_25d', 'high_ratio_25d', 'low_ratio_25d', 'turn_25d', 'pctChg_25d', 'open_ratio_28d', 'high_ratio_28d', 'low_ratio_28d', 'turn_28d', 'pctChg_28d', 'open_ratio_31d', 'high_ratio_31d', 'low_ratio_31d', 'turn_31d', 'pctChg_31d', 'open_ratio_34d', 'high_ratio_34d', 'low_ratio_34d', 'turn_34d', 'pctChg_34d', 'open_ratio_37d', 'high_ratio_37d', 'low_ratio_37d', 'turn_37d', 'pctChg_37d', 'open_ratio_40d', 'high_ratio_40d', 'low_ratio_40d', 'turn_40d', 'pctChg_40d', 'open_ratio_43d', 'high_ratio_43d', 'low_ratio_43d', 'turn_43d', 'pctChg_43d', 'open_ratio_46d', 'high_ratio_46d', 'low_ratio_46d', 'turn_46d', 'pctChg_46d', 'open_ratio_49d', 'high_ratio_49d', 'low_ratio_49d', 'turn_49d', 'pctChg_49d', 'open_ratio_52d', 'high_ratio_52d', 'low_ratio_52d', 'turn_52d', 'pctChg_52d', 'open_ratio_55d', 'high_ratio_55d', 'low_ratio_55d', 'turn_55d', 'pctChg_55d', 'open_ratio_58d', 'high_ratio_58d', 'low_ratio_58d', 'turn_58d', 'pctChg_58d', 'open_ratio_61d', 'high_ratio_61d', 'low_ratio_61d', 'turn_61d', 'pctChg_61d', 'open_ratio_64d', 'high_ratio_64d', 'low_ratio_64d', 'turn_64d', 'pctChg_64d', 'industry', 'roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare', 'NRTurnRatio', 'INVTurnRatio', 'CATurnRatio', 'AssetTurnRatio', 'YOYEquity', 'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI', 'currentRatio', 'quickRatio', 'cashRatio', 'YOYLiability', 'liabilityToAsset', 'assetToEquity', 'CAToAsset', 'tangibleAssetToAsset', 'ebitTofloaterest', 'CFOToOR', 'CFOToNP', 'CFOToGr', 'dupontROE', 'dupontAssetStoEquity', 'dupontAssetTurn', 'dupontPnitoni', 'dupontNitogr', 'dupontTaxBurden', 'dupontfloatburden', 'dupontEbittogr', 'label', 'label_1d', 'label_3d', 'label_5d', 'label_7d']
        # self.record_defaults = [['mydefault'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        # self.select_col = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176]

        self.col_name = ['date', 'code', 'rank', 'open_ratio', 'high_ratio', 'low_ratio', 'turn', 'preclose', 'amount', 'pctChg', 'peTTM', 'pcfNcfTTM', 'pbMRQ', 'isST', 'open_ratio_1d', 'high_ratio_1d', 'low_ratio_1d', 'turn_1d', 'pctChg_1d', 'open_ratio_2d', 'high_ratio_2d', 'low_ratio_2d', 'turn_2d', 'pctChg_2d', 'open_ratio_3d', 'high_ratio_3d', 'low_ratio_3d', 'turn_3d', 'pctChg_3d', 'open_ratio_4d', 'high_ratio_4d', 'low_ratio_4d', 'turn_4d', 'pctChg_4d', 'open_ratio_7d', 'high_ratio_7d', 'low_ratio_7d', 'turn_7d', 'pctChg_7d', 'open_ratio_10d', 'high_ratio_10d', 'low_ratio_10d', 'turn_10d', 'pctChg_10d', 'open_ratio_13d', 'high_ratio_13d', 'low_ratio_13d', 'turn_13d', 'pctChg_13d', 'open_ratio_16d', 'high_ratio_16d', 'low_ratio_16d', 'turn_16d', 'pctChg_16d', 'open_ratio_19d', 'high_ratio_19d', 'low_ratio_19d', 'turn_19d', 'pctChg_19d', 'open_ratio_22d', 'high_ratio_22d', 'low_ratio_22d', 'turn_22d', 'pctChg_22d', 'open_ratio_25d', 'high_ratio_25d', 'low_ratio_25d', 'turn_25d', 'pctChg_25d', 'open_ratio_28d', 'high_ratio_28d', 'low_ratio_28d', 'turn_28d', 'pctChg_28d', 'open_ratio_31d', 'high_ratio_31d', 'low_ratio_31d', 'turn_31d', 'pctChg_31d', 'open_ratio_34d', 'high_ratio_34d', 'low_ratio_34d', 'turn_34d', 'pctChg_34d', 'open_ratio_37d', 'high_ratio_37d', 'low_ratio_37d', 'turn_37d', 'pctChg_37d', 'open_ratio_40d', 'high_ratio_40d', 'low_ratio_40d', 'turn_40d', 'pctChg_40d', 'open_ratio_43d', 'high_ratio_43d', 'low_ratio_43d', 'turn_43d', 'pctChg_43d', 'open_ratio_46d', 'high_ratio_46d', 'low_ratio_46d', 'turn_46d', 'pctChg_46d', 'open_ratio_49d', 'high_ratio_49d', 'low_ratio_49d', 'turn_49d', 'pctChg_49d', 'open_ratio_52d', 'high_ratio_52d', 'low_ratio_52d', 'turn_52d', 'pctChg_52d', 'open_ratio_55d', 'high_ratio_55d', 'low_ratio_55d', 'turn_55d', 'pctChg_55d', 'open_ratio_58d', 'high_ratio_58d', 'low_ratio_58d', 'turn_58d', 'pctChg_58d', 'open_ratio_61d', 'high_ratio_61d', 'low_ratio_61d', 'turn_61d', 'pctChg_61d', 'open_ratio_64d', 'high_ratio_64d', 'low_ratio_64d', 'turn_64d', 'pctChg_64d', 'industry', 'roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare', 'NRTurnRatio', 'INVTurnRatio', 'CATurnRatio', 'AssetTurnRatio', 'YOYEquity', 'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI', 'currentRatio', 'quickRatio', 'cashRatio', 'YOYLiability', 'liabilityToAsset', 'assetToEquity', 'CAToAsset', 'tangibleAssetToAsset', 'ebitToInterest', 'CFOToOR', 'CFOToNP', 'CFOToGr', 'dupontROE', 'dupontAssetStoEquity', 'dupontAssetTurn', 'dupontPnitoni', 'dupontNitogr', 'dupontTaxBurden', 'dupontIntburden', 'dupontEbittogr', 'label', 'label_1d', 'label_3d', 'label_5d', 'label_7d']

        self.record_defaults = [['mydefault'], ['mydefault'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], ['mydefault'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        self.select_col = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.get_fea_columns()
        self.dnn_dims = self.init_variable()
        self.fea_count = sum(self.select_col) - 5

    def init_variable(self):
        dnn_hidden_1 = 1024
        dnn_hidden_2 = 512
        dnn_hidden_3 = 256
        dnn_hidden_4 = 128
        dnn_hidden_5 = 64
        dnn_hidden_6 = 32
        dnn_dims = [dnn_hidden_1, dnn_hidden_2, dnn_hidden_3, dnn_hidden_4, dnn_hidden_5, dnn_hidden_6]
        return dnn_dims

    def decode_csv(self, line):
        data = tf.decode_csv(line, record_defaults=self.record_defaults, field_delim=',', use_quote_delim=True, na_value='', name=None)
        label = data[-5:]
        features = {}
        for index in range(len(self.col_name) - len(label)):
            if self.select_col[index] == 1:
                if self.record_defaults[index][0] == 'mydefault':
                    val = tf.cast(data[index], dtype=tf.string)
                else:
                    val = tf.cast(data[index], dtype=tf.int32)
                key = self.col_name[index]
                features[key] = val
        features['label'] = label[-1]
        return features, label

    def train_input_fn_from_csv(self, data_path, epoch=1, batch_size=1024):
        with tf.device('/cpu:0'):
            dataset = tf.data.TextLineDataset(data_path).skip(1)
            dataset = dataset.shuffle(buffer_size=50000).batch(batch_size).repeat(epoch)
            dataset = dataset.map(self.decode_csv, num_parallel_calls=4).prefetch(500)
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

    def shared_embedding(self, key_name, val_name, hash_bucket):
        cate_feature = tf.feature_column.categorical_column_with_hash_bucket(key_name,hash_bucket,dtype=tf.string)
        if val_name != "":
            w_cate_feature = tf.feature_column.weighted_categorical_column(cate_feature,val_name,dtype=tf.float32)
            return w_cate_feature
        return cate_feature

    def get_fea_columns(self):
        for field_idx in range(len(self.col_name) - 5):
            if self.select_col[field_idx] == 1:
                fea_name = self.col_name[field_idx]
                # config = self.fields_config_dict[fea_name]
                # fea_type = config['type']
                # fea_field = config['field']
                if self.record_defaults[field_idx][0] == 'mydefault':
                    embeded_fea = self.hash_embedding(fea_name, 40010, self.embedding_dim)
                else:
                    embeded_fea = self.index_embedding(fea_name, 40010, self.embedding_dim)
                self.feature_columns_dict['other_feas'].append(embeded_fea)

    def model_fn_params(self):
        model_fn_params_dict = {'lr': self.lr, 'optimizer': self.optimizer, 'feature_columns': self.feature_columns_dict}
        return model_fn_params_dict

    def dnn(self, net):
        for idx, units in enumerate(self.dnn_dims):
            net = tf.layers.batch_normalization(inputs=net, name='concat_bn' + str(idx), reuse=tf.AUTO_REUSE)
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu, name='dnn_' + str(idx))
        net = tf.layers.batch_normalization(net, name="concat_bn_output")
        net = tf.layers.dense(net, units=1)
        return net

    def model_fn(self, features, labels, mode, params):
        #
        other_feas = params['feature_columns']['other_feas']

        with tf.device('/cpu:0'):
            all_feas = []
            all_feas = all_feas + other_feas
            embed_input = tf.reshape(tf.feature_column.input_layer(features, all_feas), [-1, self.embedding_dim])
            embed_input = tf.reshape(embed_input, [-1, self.embedding_dim * self.fea_count])
        with tf.device('/gpu:0'):
            y_ctr = self.dnn(embed_input)
        logging.warning('y_ctr.device: {}'.format(y_ctr.device))
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
                'predition_ctr': ctr_pred,
                'label': features['label'],
                'code': features['code']
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        # ground_truth_ctr = (labels[0, :] + labels[1, :] + labels[2, :] + labels[3, :] + labels[4, :])/5
        ground_truth_ctr1 = tf.reshape(labels[0, :], [-1])
        ground_truth_ctr2 = tf.reshape(labels[1, :], [-1])
        ground_truth_ctr3 = tf.reshape(labels[2, :], [-1])
        ground_truth_ctr4 = tf.reshape(labels[3, :], [-1])
        ground_truth_ctr5 = tf.reshape(labels[4, :], [-1])

        loss1 = tf.reduce_mean(tf.losses.log_loss(labels=ground_truth_ctr1, predictions=ctr_pred))
        loss2 = tf.reduce_mean(tf.losses.log_loss(labels=ground_truth_ctr2, predictions=ctr_pred))
        loss3 = tf.reduce_mean(tf.losses.log_loss(labels=ground_truth_ctr3, predictions=ctr_pred))
        loss4 = tf.reduce_mean(tf.losses.log_loss(labels=ground_truth_ctr4, predictions=ctr_pred))
        loss5 = tf.reduce_mean(tf.losses.log_loss(labels=ground_truth_ctr5, predictions=ctr_pred))
        loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5

        # eval
        auc_ctr = tf.metrics.auc(labels=tf.reshape(ground_truth_ctr5, [-1]), predictions=tf.reshape(ctr_pred, [-1]), name='auc_ctr_op')
        # mean_squared_error = tf.metrics.mean_squared_error(ground_truth_ctr, ctr_pred, weights=None, metrics_collections=None,updates_collections=None, name=None)
        metrics = {'auc_ctr': auc_ctr}
        tf.summary.scalar('auc_ctr', auc_ctr[1])

        # metrics = {'mean_squared_error': mean_squared_error}
        # tf.summary.scalar('mean_squared_error', mean_squared_error)
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
                input_fn=lambda: self.train_input_fn_from_csv(data_path=self.test_data, epoch=1, batch_size=self.batch_size),throttle_secs=300)

            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

            # if self.is_chief:
            #     print("......................Start savemodel......................")
            #     classifier.export_savedmodel(self.output_dir, self.serving_input_receiver_fn, strip_default_attrs=True)
        elif self.task_type == 'predict':
            print("......................Start predict......................")
            predictions = classifier.predict(
                input_fn=lambda: self.train_input_fn_from_csv(data_path=self.test_data, epoch=1,
                                                              batch_size=self.batch_size),
                predict_keys=None,
                checkpoint_path=None,
                yield_single_examples=False
            )
            try:
                while (True):
                    predict_result = next(predictions)
                    # predict_result = predict_result['predition_ctr']
                    # predict_result = predict_result['y_ctr']
                    # predict_result = predict_result['ground_truth_ctr1']
                    # predict_result = predict_result['loss']
                    print("==================")
                    print(predict_result['predition_ctr'])
                    print(predict_result['label'])
                    # print(predict_result['code'])
                    print("==================")
            except StopIteration:
                print(
                    "predict finish..., idx: {0}, time consume: {1}".format(1, 1))

        # elif self.task_type == 'savemodel':
        #     if self.is_chief:
        #         print("......................Start savemodel......................")
        #         classifier.export_savedmodel(self.output_dir, self.serving_input_receiver_fn, strip_default_attrs=True)


#
def main():
    model = DeepFM()
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
    main()
    # tf.test.is_gpu_available()
    # model = DeepFM()
    # model.run()