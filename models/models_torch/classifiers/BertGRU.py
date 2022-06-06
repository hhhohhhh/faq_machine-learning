#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/1/12 14:54 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/12 14:54   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

"""
import torch
from torch import nn
from transformers import BertModel
from conf.configs import PRETRAINED_BERT_DIR


class BertGRU(nn.Module):
    def __init__(self,pretrained_bert_dir,output_size,hidden_size,num_layers,bidirectional,dropout):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=pretrained_bert_dir)
        embedding_dim = self.bert.config.to_dict()['hidden_size']

        self.gru =  nn.GRU(input_size=embedding_dim ,hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                           batch_first=True,dropout=0 if num_layers<2 else dropout)
        self.dropout =nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=hidden_size*2 if bidirectional else hidden_size , out_features=output_size)

    def forward(self,batch):
        # text = [batch_size,len_sentence]
        # embedded = [batch_size, max_sequence_length, embedding_dim]
        with torch.no_grad():
            #  cnn 或者 bert model 的时候： batch first
            #  embedded = [ 64, 510,768]
            #  cls_embedding = [64, 768]
            embedded = self.bert(batch.text)[0]
        # output
        # hidden = (num_layers*bidirectional,batch_size, hidden_size)
        output, hidden = self.gru(embedded)
        # concate
        if self.gru.bidirectional:
            concated = torch.cat([hidden[-2,:,:], hidden[-1,:,:]],dim=1)
        else:
            concated = hidden[-1,:,:]
        dropouted = self.dropout(concated)
        output = self.linear(dropouted)
        return output



