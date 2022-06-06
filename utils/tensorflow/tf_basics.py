#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/12/6 22:30 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/6 22:30   wangfc      1.0         None
"""

import tensorflow as tf


def calculate_mean_and_variance_from_dataset(dataset:tf.data.Dataset,batch_size=8192):
    """
    计算一维的 dataset 的 平均值和 variance, 用于特征的 standarization
    只是用 reduce 方法 太慢了
    """
    # 使用 dataset.map() 方法 转换为 tf.float32
    dataset = dataset.map(map_func=lambda x: tf.cast(x, tf.float32))

    # 使用 dataset.reduce 方法计算 size
    # dataset_size = dataset.reduce(tf.cast(0,tf.float32), lambda x,_: x+1)
    # 使用 dataset.reduce 方法计算总和
    # dataset_sum  = dataset.reduce(tf.cast(0, tf.float32), lambda x, y: tf.reduce_sum([x, y]))

    dataset_size = 0
    dataset_sum = 0
    for batch in dataset.batch(batch_size):
        dataset_size += batch.shape[0]
        dataset_sum += tf.reduce_sum(batch)

    # 计算平均值
    dataset_mean = dataset_sum/dataset_size
    # 计算 variance 之和
    # dataset_variance_sum = dataset.map(lambda x: tf.square(x-dataset_mean))\
    #     .reduce(tf.cast(0,tf.float32), lambda x,y:tf.reduce_sum([x,y]))
    dataset_variance_sum = 0
    for batch in dataset.batch(batch_size=batch_size):
        dataset_variance_sum+= tf.reduce_sum(tf.square(batch - dataset_mean))
    # 计算 variance
    dataset_variance = dataset_variance_sum/dataset_size
    return dataset_mean.numpy(),dataset_variance.numpy()

def calculate_min_and_max_from_dataset(dataset:tf.data.Dataset,dtype:tf.DType=tf.int32,batch_size=8192):
    max_value = dataset.reduce(tf.cast(-1e9, dtype), tf.maximum).numpy().max()
    min_value = dataset.reduce(tf.cast(1e9,dtype), tf.minimum).numpy().min()
    return min_value ,max_value