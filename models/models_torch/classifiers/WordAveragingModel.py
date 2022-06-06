#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/1/5 13:51 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/5 13:51   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

"""
import torch
from torch.nn import Embedding,Linear
from torch.nn import AvgPool1d
import torch.nn.functional as F

class WordAveragingModel(torch.nn.Module):
    """
    我们首先介绍一个简单的Word Averaging模型。
    这个模型非常简单，我们把每个单词都通过Embedding层投射成word embedding vector，
    然后把一句话中的所有word vector做个平均，就是整个句子的vector表示了。
    接下来把这个sentence vector传入一个Linear层，做分类即可。
    """
    def __init__(self,output_size,vocab_size,padding_index,embedding_dim=100,kernel_size=1):
        super(WordAveragingModel,self).__init__()
        # 根据 vocab_size,embedding_dim 来创建 embedding层
        self.embedding = Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim,padding_idx=padding_index)
        # 使用 average pooling 创建 averaging层
        # self.pooling_averaging = AvgPool1d(kernel_size=kernel_size,stride=3)
        # 最后加一层全连接层
        self.linear = Linear(in_features=embedding_dim,out_features=output_size)

    def forward(self,batch):
        # 输入 text = [max_sequence,batch_size] -> [max_sequence,batch_size, embedding_dim]
        embedded= self.embedding(batch.text)
        #  [max_sequence,batch_size, embedding_size] ->  [batch_size,max_sequence, embedding_dim]
        embedded = embedded.permute((1,0,2))

        #  我们使用 avg_pool2d 来做average pooling。我们的目标是把sentence length那个维度平均成1，然后保留embedding这个维度。
        #  kernel_size = (max_sequence,1)
        #  input = [batch_size, max_sequence, embedding_size]
        #  [batch_size, max_sequence, embedding_dim] -> [batch size, embedding_dim]
        pooled = F.avg_pool2d(input = embedded,kernel_size = (embedded.shape[1],1)).squeeze(1)

        # 全连接层  [batch size, embedding_dim]-> [batch_size,output_size]
        output = self.linear(pooled)
        return output
