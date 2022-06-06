#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/8 11:27 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/8 11:27   wangfc      1.0         None

https://www.tensorflow.org/guide/sparse_tensor

"""
import tensorflow as tf


x = [0, 7, 0, 0, 8, 0, 0, 0, 0]
st0 = tf.SparseTensor(values=[7,8],indices=[[1],[4]],dense_shape=[9])


# Manipulating sparse tensors
# tf.sparse.add: Add sparse tensors of the same shape by using tf.sparse.add.
st_a = tf.SparseTensor(indices=[[0, 3], [2, 4]],
                      values=[10, 20],
                      dense_shape=[3, 10])


# Use tf.sparse.sparse_dense_matmul to multiply sparse tensors with dense matrices.
st_b = tf.SparseTensor(indices=[[0, 2], [7, 0]],
                       values=[56, 38],
                       dense_shape=[4, 10   ])

st_sum = tf.sparse.add(st_a, st_b)
