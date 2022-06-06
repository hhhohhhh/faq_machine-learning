#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/1/5 13:48 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/5 13:48   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

"""
import torch
from torch.nn import Linear

# 自定义的模型
class TwoLayerModel(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        # 初始化模型的所使用的layer
        super(TwoLayerModel,self).__init__()

        self.linearLayer1 = Linear(input_size,hidden_size)
        self.linearLayer2 = Linear(hidden_size,output_size)

    def forward(self,x):
        # 定义前向网络
        x = self.linearLayer1(x)
        x = self.linearLayer2(x)
        return x