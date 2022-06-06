#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/4/6 16:06 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/6 16:06   wangfc      1.0         None
"""
import torch

def test_uniform():
    size = (2,3)
    torch.rand(size=size)