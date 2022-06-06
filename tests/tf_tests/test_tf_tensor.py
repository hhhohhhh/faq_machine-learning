#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/12/17 11:37 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/17 11:37   wangfc      1.0         None


TensorFlow 支持 SparseTensor 类型，可以表示多维稀疏数据，在处理大规模数据集时会有帮助。用稀疏张量表示数据，可以压缩内存占用空间（视数据稀疏度）。

"""
import tensorflow as tf

if __name__ == '__main__':
    # indices 是一个二维 int64 类型张量，表示稀疏张量的非零元坐标；
    # values 则对应每个非零元的值；
    # dense_shape 表示转换为稠密形式后的形状信息。
    sparse_tensor = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
