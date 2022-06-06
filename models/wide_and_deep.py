#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/5/26 22:36 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/26 22:36   wangfc      1.0         None
"""
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate

class WideAndDeepModel(Model):

    def __init__(self,units=30,activation='relu',main_output_size=1,aux_output_size=1,**kwargs):
        super(WideAndDeepModel, self).__init__()
        self.units = units
        self.activation = activation
        self.main_output_size =main_output_size
        self.aux_output_size =aux_output_size

        self.hidden1 = keras.layers.Dense(units=self.units,activation=self.activation,name='hidden1')
        self.hidden2 = keras.layers.Dense(units=self.units,activation = self.activation,name='hidden2')

        self.main = keras.layers.Dense(units=self.main_output_size,name='main')
        self.aux = keras.layers.Dense(units= self.aux_output_size,name='aux')

        self.build(input_shape=(None,2,10))

    def call(self,inputs):
        wide_input = inputs[:,0]
        deep_input = inputs[:,1]
        hidden1_output = self.hidden1(deep_input)
        hidden2_output = self.hidden2(hidden1_output)
        aux_output = self.aux(hidden2_output)
        concat_output = concatenate([ hidden2_output,wide_input] )
        main_output = self.main(concat_output)
        return main_output,aux_output

if __name__ == '__main__':

    wild_and_deep_model = WideAndDeepModel()
    wild_and_deep_model.summary()

