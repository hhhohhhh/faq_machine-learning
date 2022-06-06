#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/24 17:14 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/24 17:14   wangfc      1.0         None
"""
from typing import Text, Dict, Union, Optional, NamedTuple
import  numpy  as np
import  tensorflow as tf
from rasa.utils.tensorflow.model_data import RasaModelData


class DataSignature(NamedTuple):
    """Signature of feature arrays.

    Stores the number of units, the type (sparse vs dense), and the number of
    dimensions of features.
    """

    is_sparse: bool
    units: Optional[int]
    number_of_dimensions: int


# Data = Dict[Text, tf.Tensor]


class ModelData():
    """
    参考 ： from rasa.utils.tensorflow.model_data import RasaModelData

    将数据分装为 ModelData class

    """

    def get_signature(
            self, data: Optional[Data] = None
    ) -> Dict[Text, DataSignature]:

        if not data:
            data = self.data

        return {
            key: {
                attribute_data.shape
            }
            for key, attribute_data in data.items()
        }

