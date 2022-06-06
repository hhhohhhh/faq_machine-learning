#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/14 14:12 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/14 14:12   wangfc      1.0         None
"""
import os
import sys
# sys.path.append(os.path.dirname(__file__))
from tensorflow.python.ops import math_ops, array_ops

from utils.tensorflow.environment import setup_tf_environment
setup_tf_environment(gpu_memory_config=None)

import pytest
import numpy as np
import tensorflow as tf


class MyLayer(tf.keras.layers.Layer):
    """
    Examle from
    keras.engine.base_layer.Layer()
        @losses

    """
    def call(self, inputs):
        # 对 inputs 增加 regularation
        self.add_loss(tf.abs(tf.reduce_mean(inputs)))
        return inputs


def test_inputs_regularization():

    inputs = np.ones(shape= (10, 1))
    # inputs = np.random.random(size = (4,2))
    mean = np.mean(inputs) # tf.reduce_mean(inputs)

    l = MyLayer()

    l(inputs)
    losses = l.losses
    print(f"inputs={inputs}, mean = {mean}, losses = {l.losses}")
    assert losses[0].numpy() == mean



def test_regularization():
    inputs = tf.keras.Input(shape=(10,))
    dense = tf.keras.layers.Dense(10, kernel_initializer='ones')
    x = dense(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)

    # kernel Weight regularization.
    model.add_loss(lambda :tf.abs(tf.reduce_mean(dense.kernel)))

    # activation_regularization
    model.add_loss(tf.abs(tf.reduce_mean(outputs)))
    losses = model.losses
    print(f"inputs={inputs},  losses = {losses}")
    assert losses.__len__() ==2


def test_kernel_regularizer(l1=0.01,l2=0.01):
    """
    在Keras中，我们可以方便地使用三种正则化技巧：
        keras.regularizers.l1
        keras.regularizers.l2  : `loss = l2 * reduce_sum(square(x))`
        keras.regularizers.l1_l2

    Keras中的Dense层为例，我们发现有以下三个参数：
        kernel_regularizer :对该层中的权值进行正则化，亦即对权值进行限制，使其不至于过大。
        bias_regularizer :与权值类似，限制该层中 biases 的大小
        activity_regularizer: 对该层的输出进行正则化
                计算方法：  base_layer._handle_activity_regularization(self, inputs, outputs)
                1. activity_loss = self._activity_regularizer(output)
                2. mean_activity_loss = activity_loss / batch_size

    大多数情况下，使用kernel_regularizer就足够了；
    如果你希望输入和输出是接近的，你可以使用bias_regularizer；
    如果你希望该层的输出尽量小，你应该使用activity_regularizer。

    """
    # 如何使用 regularizers
    dense = tf.keras.layers.Dense(
        5, input_dim=5,
        kernel_initializer='ones',
        kernel_regularizer=tf.keras.regularizers.L1(l1),
        activity_regularizer=tf.keras.regularizers.L2(l2))
    inputs = tf.ones(shape=(5, 5)) * 2.0
    outputs = dense(inputs)
    losses = dense.losses
    total_regularization_loss = tf.math.add_n(dense.losses)
    print(f"所有的 regularization loss ={losses},total_regularization_loss ={total_regularization_loss}")

    # keral l1_regularization loss 计算方法
    kernel_l1_loss = tf.reduce_sum(tf.abs(dense.kernel))*l1

    # activation l2 regularization: 对 output 进行 ls(平方 + 求和)+ 按照 batch_size 平均
    activity_loss = (tf.reduce_sum(tf.square(outputs)))*l2
    batch_size = math_ops.cast(
        array_ops.shape(outputs)[0], activity_loss.dtype)
    activity_l2_loss = activity_loss /batch_size
    total_loss = kernel_l1_loss + activity_l2_loss
    print(f"手动计算 keral_l1_loss ={kernel_l1_loss},activity_l2_loss={activity_l2_loss},total ={total_loss}")


    assert total_regularization_loss == l1_loss+l2_loss

if __name__ == '__main__':
    pass


