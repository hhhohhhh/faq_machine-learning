#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/29 11:07 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/29 11:07   wangfc      1.0         None
"""

import numpy as np

def np_argmax():
    x = np.array([[4, 2, 3],
                  [1, 0, 3]])
    index_array = np.argmax(x, axis=-1)
    max_values = np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1)
    max_values.squeeze(axis=-1)\


    # 沿着 axis=0
    index_array = np.argmax(x, axis=0)
    max_values = np.take_along_axis(x,np.expand_dims(index_array,axis=0), axis=0)
    print(f"x.shape={x.shape},max_values.shape={max_values.shape}")


if __name__ == '__main__':
    np_argmax()