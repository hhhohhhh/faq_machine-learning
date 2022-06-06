#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_normalization.py
@Version:  
@Desc:
@Time: 2022/6/3 20:33
@Contact :   wang_feicheng@163.com
@License :   Create by wangfc27441 on 2022/6/3, Copyright 2022 wangfc27441. All rights reserved.

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/3 20:33   wangfeicheng      1.0         None
'''
import numpy as np



def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def standardize(X):
    """ Standardize the dataset X """
    # X_std = X
    # mean = X.mean(axis=0)
    # std = X.std(axis=0)
    # for col in range(np.shape(X)[1]):
    #     if std[col]:
    #         X_std[:, col] = (XX_std:, col] - mean[col]) / std[col]
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std


