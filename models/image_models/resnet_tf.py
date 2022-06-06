#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/5/26 23:06 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/26 23:06   wangfc      1.0         None
"""
# from utils.tensorflow.environment import setup_tf_environment
# setup_tf_environment(use_cpu=False, gpu_memory_config='0:5120')
from tensorflow import keras
from tensorflow.keras.layers import Layer,Dense,Conv2D,Input,Flatten,BatchNormalization,Add,GlobalAvgPool2D
from tensorflow.keras.models import Model
from model.convolution_model_tf import ConvBlock


class CNNResidualBlock(Model):
    def __init__(self,layers,filters,**kwargs):
        super(CNNResidualBlock, self).__init__(**kwargs)

        self.hidden = [Conv2D(filters=filters,kernel_size=(3,3), activation='relu',padding='same')  for _ in range(layers)]
        # 使用自己定义的 ConvBlock ： conv + pooling
        # 但是 pooling 之后会尺度减小，如何保持尺度一致呢？
        # self.hidden = [ConvBlock(index=index, filters=filters,kernel_size=(3,3), activation='relu',padding='same')  for index in range(layers)]

    def call(self,inputs):
        x = inputs
        for layer in self.hidden:
            x =layer(x)
        # 需要保证 input 和 x 维度一样
        return inputs+x


class DNNResidualBlock(Model):
    def __init__(self,layers,units,**kwargs):
        super(DNNResidualBlock, self).__init__(**kwargs)
        self.hidden = [Dense(units=units,activation='relu') for _ in range(layers)]

    def call(self,inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x


class EasyResidualModel(Model):
    def __init__(self,input_size=(None,28,28,1) ,output_size=10):
        super(EasyResidualModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # 使用 dense 讲 shape = (None,28,28,1) -> (None,28,28,32)
        self.dense1 = Dense(32,activation='relu')
        # x = (None,28,28,32)
        self.cnn_residual_block = CNNResidualBlock(layers=2,filters=32)
        # x = (None,28,28,32)
        self.dnn_residual_block = DNNResidualBlock(layers=2,units= 32)
        #
        self.flatten  = Flatten()
        self.output_layer = Dense(1,activation='softmax')

        # self.input_layer = Input(self.input_size[1:])
        # # 可以通过在此 inputs 对象上调用层，在层计算图中创建新的节点：
        # # Get output layer with `call` method
        # self.outputs = self.call(self.input_layer)
        #
        # self.build(input_shape=self.input_size)

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.cnn_residual_block(x)
        for _ in range(3):
            x = self.dnn_residual_block(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        return x


class ConvAndBatchNorm(Model):
    def __init__(self,filters,kernel_size,padding='same',activation= 'relu',name=None):
        super(ConvAndBatchNorm, self).__init__()
        self.conv = Conv2D(filters=filters,kernel_size=kernel_size,padding=padding,name=name)
        self.batch_normalization = BatchNormalization()
        self.act = keras.activations.get(activation)

    def call(self, inputs, training=None, mask=None):
        x  = self.conv(inputs)
        x =  self.batch_normalization(x)
        x = self.act(x)
        return x



class IdentityBlock(Model):
    def __init__(self,layers=2,filters=64,kernel_size=3,padding='same',activation= 'relu',name=None):
        super(IdentityBlock, self).__init__()
        self.hidden_layers =[ConvAndBatchNorm(filters=filters,kernel_size=kernel_size,padding=padding,activation=activation)
                             for _ in range(layers)]
        self.add = Add()


    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        # skip connection
        x = self.add([x,inputs])
        return x


class MiniResidualModel(Model):
    def __init__(self,input_size=(None,28,28,1) ,output_size=10):
        super(MiniResidualModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv = Conv2D(filters=64,kernel_size=7,padding='same')
        self.bn = BatchNormalization()
        self.act =  keras.activations.get('relu')
        self.max_pooling = keras.layers.MaxPool2D()

        self.identity_blocks = [IdentityBlock() for _ in range(2)]
        self.global_avg_pooling = GlobalAvgPool2D()
        self.output_layer = Dense(units=self.output_size,activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pooling(x)
        for block in self.identity_blocks:
            x = block(x)
        x = self.global_avg_pooling(x)
        x = self.output_layer(x)
        return x



if __name__ == '__main__':
    # residual_model = EasyResidualModel()
    # residual_model.summary()
    batch =  BatchNormalization()
    batch.variables
    batch.trainable_variables


