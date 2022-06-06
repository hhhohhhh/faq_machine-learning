#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/14 16:35 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/14 16:35   wangfc      1.0         None

tf.function works by tracing the Python code to generate a ConcreteFunction (a callable wrapper around tf.Graph)
When saving a tf.function, you're really saving the tf.function's cache of ConcreteFunctions.


"""
import tensorflow as tf


class F():
    def __init__(self):
        self._b = None

    @tf.function
    def __call__(self):
        a = tf.constant([[10, 10], [11., 1.]])
        x = tf.constant([[1., 0.], [0., 1.]])
        if self._b is None:
            self._b = tf.Variable(12.)
        y = tf.matmul(a, x) + self._b
        print("PRINT: ", y)
        tf.print("TF-PRINT: ", y)
        return y