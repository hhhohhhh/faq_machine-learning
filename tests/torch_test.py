#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/3/2 11:10 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/2 11:10   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

"""

import numpy  as np
import torch
from torch import nn
import torch.nn.functional as F

batch_size =2
max_seq=10
cand_num=3
codes_n =4
embedding_size =5

context_reps = torch.range(0,batch_size*cand_num*embedding_size-1).view((batch_size,cand_num,embedding_size))

cand_reps = torch.range(0,batch_size*cand_num*embedding_size-1).view((batch_size,cand_num,embedding_size))
constant =torch.Tensor([10,10,10,10,10]).unsqueeze(dim=0).repeat(2,1)

cand_reps= cand_reps.div(10)

scores =(context_reps* cand_reps).sum(dim=-1)
F.softmax(scores,dim=-1,dtype=torch.float)

# 初始化 codes
codes = torch.empty((codes_n,embedding_size))
codes = torch.nn.init.uniform_(codes)
codes = torch.nn.Parameter(codes)


#
new_ones =context_reps.new_ones(batch_size,codes_n)
new_ones.byte()

anchor = np.arange(batch_size*max_seq).reshape(batch_size,max_seq)
positive = np.arange(batch_size*max_seq,batch_size*max_seq*2).reshape(batch_size,max_seq)
negative = - anchor

# 对 anchor，positive，negative 进行拼接: (batch_size,3*max_seq)
concated = np.concatenate([anchor,positive,negative],axis=1)
# 转换为 3*batch 输入模型进行训练: (batch_size*3,max_seq)
concated_batch = concated.reshape(-1,max_seq)

# 对输出进行 reshape： (batch_size,3,embedding_size)
output_reshape = concated_batch.reshape(-1,3,max_seq)
anchor_embedding = output_reshape[:,0,:]
positive_embedding = output_reshape[:,1,:]
neg_embedding = output_reshape[:,2,:]

