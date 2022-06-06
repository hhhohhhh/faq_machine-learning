#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:




Course 1: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
Week 1: A New Programming Paradigm
The “Hello World” of neural networks

Ex:
x= -1,1,2,3,4,5
y=--3,1,3,5,7,9





@time: 2021/5/19 18:08 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/19 18:08   wangfc      1.0         None
"""
import tensorflow as tf
from .temp_keras_modules import TmpKerasModel
from tensorflow import keras



class LinearModel(TmpKerasModel):
    def __init__(self,input_size=(1,),hidden_units=1,hidden_activation=None,output_size=1,output_activation=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.output_size = output_size

        self.hidden_activation =None
        self.output_activation =None
        if hidden_activation is not None and hidden_activation=='relu':
            self.hidden_activation = tf.nn.relu
        if output_activation is not None and output_activation == 'softmax':
            self.output_activation = tf.nn.softmax

        self.flatten = keras.layers.Flatten(input_shape=self.input_size)
        self.hidden_layer =  keras.layers.Dense(units=self.hidden_units,activation=self.hidden_activation)
        self.output_layer = keras.layers.Dense(units=self.output_size,activation=self.output_activation)

        # Sequential 的方式定义模型
        # self.model = keras.Sequential(
        #     [
        #         keras.layers.Flatten(input_shape=self.input_size),
        #         keras.layers.Dense(units=self.hidden_units,activation=self.hidden_activation),
        #         keras.layers.Dense(units=self.output_size,activation=self.output_activation)
        # ])

        # model.call(inputs=inputs)


    def call(self,inputs):
        x= self.flatten(inputs=inputs)
        x= self.hidden_layer(x)
        x= self.output_layer(x)
        return x



