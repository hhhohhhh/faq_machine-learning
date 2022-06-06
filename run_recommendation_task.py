#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/9 23:27 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/9 23:27   wangfc      1.0         None
"""
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def run_movie_recommendation_task(debug=False):
    from utils.tensorflow.environment import setup_tf_environment
    setup_tf_environment(gpu_memory_config='1:5120')
    from tasks.recommendation_task import MovielensRecommendationTask
    movielens_recommendation_task = MovielensRecommendationTask(
        task='multitask',
        model_name='MultiTaskRecommendationModel',  # 'TwoTowerRankingModel',
        use_retrieval_task=True,
        use_ranking_task=True,
        retrieval_weight=1.0,
        ranking_weight=1.0,
        tower_type="DCN",
        hidden_layer_sizes=[64, 32],
        train_epochs=1,
        optimizer_name='adagrad',
        learning_rate=0.1,
        debug=debug)
    mode = 'train'
    if mode == 'train':
        movielens_recommendation_task.train()
    elif mode == 'infer':
        movielens_recommendation_task.load()
        result = movielens_recommendation_task.infer(inputs={'user_id': tf.constant(['42'])})
        # print(f"result= {result}")


def run_criteo_classification_example():
    import tensorflow as tf

    from tensorflow.python.ops.parsing_ops import FixedLenFeature
    from deepctr.estimator.inputs import input_fn_tfrecord
    from deepctr.estimator.models import DeepFMEstimator

    # Step 2: Generate feature columns for linear part and dnn part
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    dnn_feature_columns = []
    linear_feature_columns = []

    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feat, 1000), 4))
        linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, 1000))
    for feat in dense_features:
        dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
        linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    # Step 3: Generate the training samples with TFRecord format
    feature_description = {k: FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features}
    feature_description.update(
        {k: FixedLenFeature(dtype=tf.float32, shape=1) for k in dense_features})
    feature_description['label'] = FixedLenFeature(dtype=tf.float32, shape=1)

    train_model_input = input_fn_tfrecord('./criteo_sample.tr.tfrecords', feature_description, 'label', batch_size=256,
                                          num_epochs=1, shuffle_factor=10)
    test_model_input = input_fn_tfrecord('./criteo_sample.te.tfrecords', feature_description, 'label',
                                         batch_size=2 ** 14, num_epochs=1, shuffle_factor=0)

    # Step 4: Train and evaluate the model
    model = DeepFMEstimator(linear_feature_columns, dnn_feature_columns, task='binary')

    model.train(train_model_input)
    eval_result = model.evaluate(test_model_input)

    print(eval_result)


def run_movie_ctr_task():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.python.keras.preprocessing.sequence import pad_sequences

    from deepctr.models import DeepFM
    from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    data = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip", ]
    target = ['rating']

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature

    key2index = {}
    genres_list = list(map(split, data['genres'].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)
    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4)
                              for feat in sparse_features]

    use_weighted_sequence = False
    if use_weighted_sequence:
        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
            key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                                   weight_name='genres_weight')]
        # Notice : value 0 is for padding for sequence input feature
    else:
        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
            key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                                   weight_name=None)]
        # Notice : value 0 is for padding for sequence input feature

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    model_input = {name: data[name] for name in feature_names}  #
    model_input["genres"] = genres_list
    model_input["genres_weight"] = np.random.randn(data.shape[0], max_len, 1)

    # 4.Define Model,compile and train
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')

    model.compile("adam", "mse", metrics=['mse'], )
    history = model.fit(model_input, data[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )


if __name__ == '__main__':
    run_movie_recommendation_task(debug=True)
