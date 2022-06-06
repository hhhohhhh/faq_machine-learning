#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/1/6 16:40 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/6 16:40   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

"""
import logging

logger = logging.getLogger(__name__)

import torch
from torch import nn
from utils.utils import count_parameters


class RNN(nn.Module):
    def __init__(self, output_size, vocab_size, embedding_dim, padding_index, hidden_size, num_layers, bidirectional):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_index)

        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional)

        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

        logger.info(f'The rnn model has {count_parameters(self.rnn)} trainable parameters')

    def forward(self, batch):
        # text = [max_seq,batch_size]
        # embedded = [max_seq,batch_size, embedding_dim]
        embedded = self.embedding(batch.text)

        # output = (max_seq, batch_size,hidden_size), hidden = (1,batch_size,hidden_dim)
        # 单层单向的时候
        # input= (seq_len, batch, input_size)
        # output = (seq_len, batch, output_size) ,hidden = [1, batch size, hidden dim].
        # 公式： tanh(Ux + Wh+ b)
        # 权重数量： hidden_size * embedding_dim + hidden_size+ hidden_size * hidden_size + hidden_size

        # 多层单向的时候
        # input= (seq_len, batch, input_size)
        # output = (seq_len, batch, hidden_size * bidirectional) ,hidden = [num_layers*bidirectional, batch size, hidden dim].
        # 公式： tanh(Ux + Wh+ b)
        # 权重数量： (hidden_size * embedding_dim + hidden_size+ hidden_size * hidden_size + hidden_size) * bidirectional * num_layers

        output, hidden = self.rnn(input=embedded)

        predict = self.linear(hidden.squeeze(dim=0))
        return predict


class LSTM(nn.Module):
    def __init__(self, output_size, vocab_size, embedding_dim, padding_index,
                 hidden_size, num_layers, bidirectional, dropout):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_index)
        # input= (seq_len, batch, input_size)
        # output = (seq_len, batch, output_size) ,hidden = [1, batch size, hidden dim].
        # 公式： tanh(Ux + Wh+ b)
        # 权重数量： (hidden_size * embedding_dim + hidden_size+ hidden_size * hidden_size + hidden_size)*4
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)
        # assert count_parameters(self.lstm)== (hidden_size * embedding_dim + hidden_size + hidden_size * hidden_size + hidden_size) * 4
        if bidirectional:
            in_features = 2 * hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features=in_features, out_features=output_size)

    def forward(self, batch):
        # text = [max_seq,batch_size]
        # embedded = [max_seq,batch_size, embedding_dim]
        text, text_lengths = batch.text
        embedded = self.embedding(text)
        # 增加 pack部分
        packed_embedded = nn.utils.rnn.pack_padded_sequence(input=embedded, lengths=text_lengths)
        # output = (max_seq, batch_size,hidden_size),
        # hidden = (num_layers * num_directions,batch_size,hidden_dim)
        # cell = (num_layers * num_directions,batch_size,hidden_dim)

        packed_output, (hidden, cell) = self.lstm(input=packed_embedded)
        # 增加 unpack 部分
        # unpack_paddend_sequence = nn.utils.rnn.pad_packed_sequence(hidden)

        # hidden_cated = ( num_directions,batch_size,hidden_dim)
        hidden_cated = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)

        dropped = self.dropout(hidden_cated)

        predict = self.linear(dropped)
        return predict
