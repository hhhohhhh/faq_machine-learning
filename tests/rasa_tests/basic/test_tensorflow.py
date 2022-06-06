#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2020/11/9 14:02 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/9 14:02   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

"""
import tensorflow as tf
import numpy as np


a= tf.reshape(tf.range(6),(3,2))
tf.reduce_sum(a)
tf.reduce_sum(a, axis=1)
tf.reduce_sum(a,axis=0)

tf.reduce_max(a,axis=1)

b= tf.reshape(tf.range(10,16),(3,2))
print(b)

tf.concat([a,b],axis=-1)

c = tf.reshape(tf.random.shuffle(tf.range(6)),shape=(3,2))
tf.reduce_mean(tf.cast(tf.math.equal(a,c),tf.float32))


class F():
    def __init__(self):
        self._b = None

    @tf.function
    def __call__(self):
        a = tf.constant([[10, 10], [11., 1.]])
        x = tf.constant([[1., 0.], [0., 1.]])
        if self._b is None:
            self._b = tf.Variable(12.)
        y = tf.matmul(a, x) + self._b
        print("PRINT: ", y)
        tf.print("TF-PRINT: ", y)
        return y

f = F()

f()

# 抽样
x = tf.reshape(tf.range(32),(32,1))
a= tf.reduce_sum(x, axis=1)
print(a.shape)
b= self._tf_layers['ffnn.label'](a)
print(b.shape)

# a = (batch_size, target_size, emb_size)
a = tf.reshape(tf.range(24),(2,6,2))
print(a)
# b= (batch_size,sample_index)
b = np.array([[0,0],
             [0,1],
              [1,1],
              [0,1]])
c = tf.gather(a,b,batch_dims=1)
print(c.shape)
print(c)

# 抽样
population_size=3
emb_size=5
batch_size =2
x = tf.reshape(tf.range(population_size*emb_size),(population_size,emb_size))
print(x.shape)
tiled = tf.tile(tf.expand_dims(x, 0), (batch_size, 1, 1))
print(tiled.shape)

neg_num=2
idxs = np.random.randint(0,2,size=(batch_size,neg_num))
tf.gather(tiled, idxs, batch_dims=1)


# 如何使用 regularizers
layer = tf.keras.layers.Dense(
    5, input_dim=5,
    kernel_initializer='ones',
    kernel_regularizer=tf.keras.regularizers.L1(0.01),
    activity_regularizer=tf.keras.regularizers.L2(0.01))
tensor = tf.ones(shape=(5, 5)) * 2.0
out = layer(tensor)
layer.losses
tf.math.add_n(layer.losses)
