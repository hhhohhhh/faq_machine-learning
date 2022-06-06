#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/5/20 16:40 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/20 16:40   wangfc      1.0         None
"""
import tensorflow as tf
from tensorflow import keras
from .temp_keras_modules import TmpKerasModel






class BaiscConvolutionModel(TmpKerasModel):
    def __init__(self,input_size,filters_01=32,kernel_size= (3,3),pooling='max',
                 filters_02=128,dense_units = 512,
                 output_size=10,output_activation = 'softmax',**kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.filters_01 = filters_01
        self.filters_02= filters_02
        self.kernel_size =kernel_size
        self.dense_units = dense_units
        self.output_size = output_size

        self.conv_01 = keras.layers.Conv2D(input_shape=input_size,
                                           filters=filters_01,kernel_size=kernel_size,
                                           activation= tf.nn.relu,
                       name='conv_01')
        if pooling == 'max':
            self.pooling_01 = keras.layers.MaxPool2D()

        elif pooling == 'avg':
            self.pooling_01 = keras.layers.AvgPool2D()

        self.conv_02 = keras.layers.Conv2D(filters=filters_02, kernel_size=kernel_size,
                                           activation=tf.nn.relu,
                                        name='conv_02')
        self.pooling_02 = keras.layers.MaxPool2D()

        self.flatten = keras.layers.Flatten()
        self.dense_01 = tf.keras.layers.Dense(self.dense_units, activation='relu')

        if output_activation == 'softmax':
            self.output_activation = tf.nn.softmax
        elif output_activation == 'sigmoid':
            self.output_activation =  tf.nn.sigmoid

        self.dense_02 = keras.layers.Dense(units=output_size,activation=self.output_activation)

        # Functional API：
        # 先创建一个输入节点 ： Add input layer
        self.input_layer = keras.layers.Input(input_size[1:])
        # 可以通过在此 inputs 对象上调用层，在层计算图中创建新的节点：
        # Get output layer with `call` method
        self.outputs = self.call(self.input_layer)
        # 可以通过在层计算图中指定模型的输入和输出来创建 Model：
        # model = Model(inputs=inputs, outputs=outputs, name="mnist_model")

        # Reinitial
        # super(BaiscConvolutionModel,self).__init__(
        #     inputs=self.input_layer,
        #     outputs=self.outputs,**kwargs)

        # `input_shape` is the shape of the input data e.g. input_shape = (None, 32, 32, 3)
        self.build(input_shape=input_size)



    def call(self,inputs):
        x = self.conv_01(inputs)
        x = self.pooling_01(x)
        x = self.conv_02(x)
        x = self.pooling_02(x)
        x = self.flatten(x)
        x = self.dense_01(x)
        x = self.dense_02(x)
        return x

    # def model(self):
    #     x = Input(shape=(24, 24, 3))
    #     return Model(inputs=, outputs=self.call(x))



def conv3x3(filters, input_shape=None,kernel_size=(3, 3), activation=tf.nn.relu,padding= 'same',
             name = None):
    if input_shape is not None:
        conv = keras.layers.Conv2D(
            input_shape=input_shape,
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            name=name,
        )
    else:
        conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding = padding,
            name=name)
    return conv

def get_pooling(pooling_name,padding):
    if pooling_name == 'max':
        pooling = keras.layers.MaxPool2D(padding=padding)

    elif pooling_name == 'avg':
        pooling = keras.layers.AvgPool2D(padding=padding)
    return pooling



class ConvBlock(keras.models.Model):
    def __init__(self,index ,filters, input_size=None,kernel_size=(3, 3), activation=tf.nn.relu,
                 padding='same',
                 pooling_name='max'):
        super(ConvBlock, self).__init__()
        self.index = index
        self.filters = filters
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.pooling_name = pooling_name
        self.conv = conv3x3(filters=self.filters,
                            input_shape=self.input_size,
                            kernel_size=self.kernel_size,
                            activation=self.activation,
                            padding=self.padding,
                            name = f"conv_{self.index}")
        self.pooling = get_pooling(self.pooling_name,padding=self.padding)

    def call(self,x):
        x = self.conv(x)
        x = self.pooling(x)
        return x



class ThreeConvolutionLayerModelV1(TmpKerasModel):
    def __init__(self,input_size,filters_ls=[16,32,64],kernel_size= (3,3),pooling_name='max',
                 dense_units = [512],
                 output_size=1,output_activation = 'sigmoid'):
        super().__init__()
        self.input_size = input_size
        # filters_ls 从小到大的时候，似乎对于 dogs_and_cats 的数据集收敛性更好
        self.filters_ls = filters_ls

        self.kernel_size =kernel_size
        self.pooling_name = pooling_name
        self.dense_units = dense_units
        self.output_size = output_size
        self.output_activation = output_activation



        self.conv_ls = []
        self.pooling_ls= []
        for index,filters in enumerate(self.filters_ls):
            name = f'conv_{index}'
            if index ==0:
                input_size = self.input_size
                conv = conv3x3(filters,input_shape=input_size,kernel_size=self.kernel_size, name=name)
            else:
                conv = conv3x3(filters, kernel_size=self.kernel_size, name=name)

            self.conv_ls.append(conv)
            self.pooling_ls.append(get_pooling(self.pooling_name))

        self.flatten = keras.layers.Flatten()

        self.dense_ls = []
        self.dense_ls = [keras.layers.Dense(units=units) for units in self.dense_units]
        self.output_layer = keras.layers.Dense(units= self.output_size,activation=self.output_activation)


        # Add input layer
        self.input_layer = keras.layers.Input(input_size[1:])

        # Get output layer with `call` method
        self.outputs = self.call(self.input_layer)

        self.build(input_shape=input_size)



    def call(self,input):
        x = input
        for conv,pooling in zip(self.conv_ls,self.pooling_ls):
            x = conv(x)
            x = pooling(x)
        x = self.flatten(x)
        for dense in self.dense_ls:
            x = dense(x)
        x = self.output_layer(x)
        return x



class ThreeConvolutionLayerModelV2(TmpKerasModel):
    def __init__(self,input_size,filters_ls=[16,32,64],kernel_size= (3,3),pooling_name='max',
                 dense_units = [512],
                 output_size=1,output_activation = 'sigmoid'):
        super().__init__()
        self.input_size = input_size
        # filters_ls 从小到大的时候，似乎对于 dogs_and_cats 的数据集收敛性更好
        self.filters_ls = filters_ls

        self.kernel_size =kernel_size
        self.pooling_name = pooling_name
        self.dense_units = dense_units
        self.output_size = output_size
        self.output_activation = output_activation

        self.conv_blocks = []
        for index,filters in enumerate(self.filters_ls):
            input_size =None
            if index ==0:
                input_size = self.input_size
            conv_block = ConvBlock(index=index,filters=filters,input_size=input_size,pooling_name=self.pooling_name)
            self.conv_blocks.append(conv_block)


        self.flatten = keras.layers.Flatten()

        self.dense_ls = []
        self.dense_ls = [keras.layers.Dense(units=units,name=f'dense_{index}') for index,units in enumerate(self.dense_units)]
        self.output_layer = keras.layers.Dense(units= self.output_size,activation=self.output_activation)


        # Add input layer
        self.input_layer = keras.layers.Input(self.input_size[1:])

        # Get output layer with `call` method
        self.outputs = self.call(self.input_layer)

        self.build(input_shape=self.input_size)



    def call(self,inputs):
        x = inputs
        for conv in self.conv_blocks:
            x = conv(x)
        x = self.flatten(x)
        for dense in self.dense_ls:
            x = dense(x)
        x = self.output_layer(x)
        return x





