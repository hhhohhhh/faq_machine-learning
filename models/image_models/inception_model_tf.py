#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/5/23 21:00 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/23 21:00   wangfc      1.0         None
"""

from .temp_keras_modules import TmpKerasModel
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3


import logging
logger = logging.getLogger(__name__)


class InceptionModel(Model):
    def __init__(self,input_size, pretrained_model_path,pretrained_layer_trainable):
        super(InceptionModel,self).__init__()
        self.input_size = input_size
        self.pretrained_model_path = pretrained_model_path
        self.pretrained_layer_trainable = pretrained_layer_trainable
        self.inception = self.load_pretrained_model()

        # 初始化 inception 预训练模型
        self.inception = self.load_pretrained_model()
        # 定义 inception 输出的 layer
        self.inception_output_layer = self.inception.get_layer('mixed7')
        # 定义 预训练模型的输入和输出
        self.pretrained_inception = Model(inputs= self.inception.input,outputs = self.inception_output_layer.output)


    def call(self, inputs, training=None, mask=None):
        return self.pretrained_inception.call(inputs)

    def load_pretrained_model(self) -> InceptionV3:
        if self.input_size.__len__() == 4:
            input_shape = self.input_size[1:]
        else:
            input_shape = self.input_size
        inception = InceptionV3(input_shape=input_shape, include_top=False, weights=None)
        inception.load_weights(self.pretrained_model_path)
        for layer in inception.layers:
            layer.trainable = self.pretrained_layer_trainable
        logger.info(
            f"加载预训练模型 {inception.__class__.__name__} from {self.pretrained_model_path}, layer.trainabel = {self.pretrained_layer_trainable}")
        return inception



class TransferInceptionModel(TmpKerasModel):
    def __init__(self,input_size,output_size,
                 pretrained_model_path='/home/wangfc/event_extract/tutorial/pretrained_model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                 pretrained_layer_trainable=False,
                 dense_units=1024,activation = 'relu',
                 dropout_rate= 0.2,output_activation = 'sigmoid'):
        super(TransferInceptionModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.pretrained_model_path = pretrained_model_path
        self.pretrained_layer_trainable = pretrained_layer_trainable
        self.dense_units = dense_units
        self.activation = activation
        self.dropout_rate =dropout_rate
        self.output_activation = output_activation

        self.pretrained_inception = InceptionModel(input_size=self.input_size,pretrained_model_path=self.pretrained_model_path,
                                                   pretrained_layer_trainable=self.pretrained_layer_trainable)

        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(units= self.dense_units,activation=self.activation)
        self.dropout = keras.layers.Dropout(rate = self.dropout_rate)
        self.output_layer = keras.layers.Dense(units=self.output_size,activation =self.output_activation)

        # Add input layer
        self.input_layer = keras.layers.Input(input_size[1:])

        # Get output layer with `call` method
        self.outputs = self.call(self.input_layer)

        self.build(input_shape=self.input_size)






    def call(self,inputs):
        x = self.pretrained_inception(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        x= self.dropout(x)
        x =self.output_layer(x)
        return x




