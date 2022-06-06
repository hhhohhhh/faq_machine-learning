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

# 定义前向转播的模型
model = Sequential(Linear(in_features=input_size, out_features=hidden_size, bias=True),
                   ReLU(),
                   Linear(in_features=hidden_size, out_features=output_size, bias=True))

# 对模型中的参数进行初始化
# torch.nn.init.xavier_normal_(model[0])
# torch.nn.init.xavier_normal_(model[1])

# 定义损失函数
loss_fn =torch.nn.CrossEntropyLoss(reduction="sum")

# 定义 优化器
learning_rate = 5e-5
optimizer = torch.optim.Adam(params=model.parameters(), lr= learning_rate )

# 创建前向训练过程
steps =1000

for step in range(steps):
    # 定义 loss
    output = model(x)
    loss = loss_fn(input=output,target=y)
    print("step={},loss={}".format(step,loss.item()))

    # 更新参数
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

