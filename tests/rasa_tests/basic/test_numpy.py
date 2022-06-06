#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2020/11/8 11:05 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/8 11:05   wangfc      1.0         None

"""
import numpy as np
from rasa.utils.tensorflow.model_data import RasaModelData

def test_rasa_model_data():
    # 多标签的情况
    label_ids = np.random.randint(low=0,high=4,size=(4,2))
    print(label_ids)
    label_ids_exp = np.expand_dims(label_ids,axis=-1)
    print(label_ids_exp.shape)
    RasaModelData._create_label_ids(label_ids=label_ids_exp )

    # 单标签的情况
    label_ids_ls =  label_ids[:,0].tolist()
    print(label_ids_ls)
    np.array(label_ids_ls)
    label_ids_ls_exp = np.expand_dims(label_ids_ls,axis=-1)
    print(label_ids_ls_exp.shape) # shape = (batch_size, 1)
    print(label_ids_ls_exp.ndim)
    RasaModelData._create_label_ids(label_ids=label_ids_ls_exp )
    print(label_ids_ls_exp[:,0])

    a = np.random.randint(0,2,size=(3,10))
    a.any(axis=0).astype(int)
