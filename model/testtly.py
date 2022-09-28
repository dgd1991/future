import numpy as np
import tensorflow as tf

#
# input_data = np.arange(16)
#
#
# dataset = tf.data.Dataset.from_tensor_slices(
#     input_data  # numpy数组或者list列表
# )  # numpy数组转tf.Tensor数组（即：TensorSliceDataset）
#
# datasets = dataset.shuffle(buffer_size=10) # 把dataset打乱顺序
#
# for data in datasets:
#     print(data)
#



#
# features = {"tokens":[["4","5","6"],["4","5","6"],["4","5","6"]],"user_id":["4","6","7"],"item_id":["3","6","8"],"map_id":[["4","5","6"],["4","5","6"],["4","5","6"]],"map_w":[[1.0,0.000001,1.0],[1.5,0.01,1.0],[1.5,0.01,1.0]]}
# all_feas = {"to":[["4","5","6"],["4","5","6"],["4","5","6"]],"user":["4","6","7"],"item":["3","6","8"],"map":[["4","5","6"],["4","5","6"],["4","5","6"]],"map_uu":[[1.0,0.000001,1.0],[1.5,0.01,1.0],[1.5,0.01,1.0]]}
#
# dense_tensor = tf.feature_column.input_layer(features,all_feas )
# with tf.Session() as session:
# 	session.run(tf.global_variables_initializer())
# 	print(session.run(dense_tensor))
# #
# embed_input = tf.reshape(tf.feature_column.input_layer(features, all_feas), [-1, self.embedding_dim])
# embed_input = tf.reshape(embed_input, [-1, self.embedding_dim * self.fea_count])


# import tensorflow as tf
# sess=tf.Session()
# features = {
#     'department': ['sport', 'sport', 'drawing', 'gardening', 'travelling'],
# }
# # 特征列
# department = tf.feature_column.categorical_column_with_hash_bucket('department', 4, dtype=tf.string)
# department = tf.feature_column.indicator_column(department)
#
# #组合特征列
# columns = [
#     department
# ]
#
# #输入层（数据，特征列）
# inputs = tf.feature_column.input_layer(features, columns)
#
# #初始化并运行
# init = tf.global_variables_initializer()
# sess.run(tf.tables_initializer())
# sess.run(init)
#
# v=sess.run(inputs)
# print(v)

#
# import tensorflow as tf
# sess=tf.Session()
#
# name = 'code'
# hash_bucket = 5
# dim = 6
# cate_feature = tf.feature_column.categorical_column_with_hash_bucket(name,
#                                                                           hash_bucket,
#                                                                           dtype=tf.string)
#
# emb_col = tf.feature_column.embedding_column(
#     cate_feature,
#     dimension=dim,
#     combiner='mean',initializer=tf.truncated_normal_initializer(stddev=0.1)
# )
# indicator_column = tf.feature_column.indicator_column(cate_feature)
#
# fea = ['a|b|c', 'C|n']
# sparse_tensor = tf.string_split(fea, delimiter="|", skip_empty=False)
# sparse_tensor_1 = tf.sparse_tensor_to_dense(sparse_tensor, default_value="", validate_indices=True)
# nafe = {'code':[['1','5'],['2','3'],['4','2']]}
# dense_tensor = tf.feature_column.input_layer(nafe,indicator_column)
# with tf.Session() as session:
# 	session.run(tf.global_variables_initializer())
# 	# print(session.run(dense_tensor))
# 	print(session.run(dense_tensor))
# 	print(session.run(tf.reshape(dense_tensor, [-1])))

#
# key_name= 'name'
# dim = 4
# hash_bucket = 4
# id_feature = tf.feature_column.categorical_column_with_identity(key_name, num_buckets=hash_bucket, default_value=0)
# emb_col = tf.feature_column.embedding_column(id_feature, dim,
#                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
# # ind_col = feature_column.indicator_column(id_feature)
# import tensorflow as tf
# sess=tf.Session()
# list = []
# nafe = {'name': [1,3,9],'age':['1','3','0']}
# fea_nzme = ["name", "age"]
#
# for fea in fea_nzme:
#     if fea == 'name':
#         id_feature = tf.feature_column.categorical_column_with_identity(fea, num_buckets=hash_bucket, default_value=0)
#         emb_col = tf.feature_column.embedding_column(id_feature, dim,
#                                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
#         list.append(emb_col)
#     elif  fea == 'age':
#         cate_feature = tf.feature_column.categorical_column_with_hash_bucket(fea,
#                                                                                   hash_bucket,
#                                                                                   dtype=tf.string)
#
#         emb_col = tf.feature_column.embedding_column(
#             cate_feature,
#             dimension=dim,
#             combiner='mean',initializer=tf.truncated_normal_initializer(stddev=0.1)
#         )
#         list.append(emb_col)
#
#
#
# naf = list
# dense_tensor  =  tf.feature_column.input_layer(nafe ,naf)
# with tf.Session() as session:
#     session.run(tf.global_variables_initializer())
#     print(session.run(dense_tensor))
#



# key_name = 'code'
# hash_bucket = 10
# val_name = 'names'
# cate_feature = tf.feature_column.categorical_column_with_hash_bucket(key_name,hash_bucket,dtype=tf.string)
# if val_name != "":
#     w_cate_feature = tf.feature_column.weighted_categorical_column(cate_feature,val_name,dtype=tf.float32)
#
# dicts = {'other_feas': []}
# dicts['other_feas'].append('1')
#
# print(dicts)
#

l = [2,3,4]
sums = 0
for  i  in l :
    sums +=i
sums+=i
print(sums)