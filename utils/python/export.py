#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/15 9:03 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/15 9:03   wangfc      1.0         None
"""
import functools

class APIExport():
    """
    参考 @tf_export 的方法
    """
    def __init__(self,api_name):
        self.api_name = api_name


    def __call__(self, *args, **kwargs):
        pass


# api_export = functools.partial()