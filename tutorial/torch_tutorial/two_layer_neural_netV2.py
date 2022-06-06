#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/1/4 14:25 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/4 14:25   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

"""

from __future__ import print_function
import torch
from torch.nn import Sequential,Linear,ReLU
from utils import get_parameter_number
from model.two_layer_model import TwoLayerModel
# 创建数据
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
batch_size=64
input_size= 128
hidden_size = 100
output_size = 2

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
device = torch.device("cpu")
dtype = torch.float

# Create random input and output data
x = torch.randn(batch_size, input_size, dtype=dtype)
y = torch.randint(low=0,high=2,size=(batch_size,))


# 使用自定义的模型类来构建 model
model = TwoLayerModel(input_size=input_size,hidden_size=hidden_size,output_size=output_size)


# 定义 loss 函数
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

# 定义 优化器
learning_rate = 5e-5
optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)

steps = 1000

for step in range(steps):
    output = model(x)
    loss = loss_fn(output,y)
    print("step={},loss={}".format(step,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()






