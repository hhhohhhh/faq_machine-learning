#!/usr/bin/env python
# coding: utf-8

# # 第六课 Seq2Seq, Attention

import os
import sys
import math
from collections import Counter
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
# nltk.download('punkt')
from Sequence_to_Sequence_Learning_with_Neural_Networks import *


# ### 没有Attention的版本
# 下面是一个更简单的没有Attention的encoder decoder模型

class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # 对输入的数据进行排序，如果已经排序，其实不需要这一步
        # sorted_len, sorted_idx = lengths.sort(0, descending=True)
        # x_sorted = x[sorted_idx.long()]
        # embedded = self.dropout(self.embed(x_sorted))
        # packed_embedded = pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)
        embedded = self.dropout(self.embed(x))
        # 在encoder中的rnn只计算到 non pad的那个位置，需要先对输入的进行 pack
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)

        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        # out = [batch_size, seq, hid_dim] , hid = [n_layers * n_dirction, batch_size, hid_dim]
        # print(f'encoder out.shape:{out.shape},hid.shape:{hid.shape}')
        # _, original_idx = sorted_idx.sort(0, descending=False)
        # out = out[original_idx.long()].contiguous()
        # hid = hid[:, original_idx.long()].contiguous()
        # 因为 hid的shape和batch_first 无关
        return out, hid[[-1]]


class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, y_lengths, hid):
        # 因为原来的数据是按照 src的句子的长短来排序的，这儿在encoder端，如果 enforce_sorted = True，需要对trg的语句 进行排序
        # sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        # y_sorted = y[sorted_idx.long()]
        # hid = hid[:, sorted_idx.long()]
        # y_sorted = self.dropout(self.embed(y_sorted)) # batch_size, output_length, embed_size
        # packed_seq = pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)

        y = self.dropout(self.embed(y))
        # 在decoder中的rnn只计算到 non pad的那个位置，需要先对输入的进行 pack
        packed_seq = pack_padded_sequence(y, y_lengths, batch_first=True,
                                          enforce_sorted=False)  # y = [batch_size,seq_len, emb_dim]
        # 在decoder中： 将encoder 的 hidden state 作为 decoder中的 初始的 h_0
        out, hid = self.rnn(packed_seq, hid)

        unpacked, unpacked_lengths = pad_packed_sequence(out, batch_first=True)
        # out = [batch_size, seq_len, hidden_dim * n_direction]
        # hid = [ n_direction*layers,batch_size, hidden_dim]
        # print(f'decoder unpacked.shape:{unpacked.shape},hid.shape:{hid.shape}')
        # _, original_idx = sorted_idx.sort(0, descending=False)
        # output_seq = unpacked[original_idx.long()].contiguous()
        # print(output_seq.shape)
        # hid = hid[:, original_idx.long()].contiguous()
        output = F.log_softmax(self.out(unpacked), -1)  # out = [batch_size, seq_len, vocab_size]
        # print(f'decoder output.shape:{output.shape},hid.shape:{hid.shape}')
        return output, hid


class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(PlainSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)  # x = [batch_size,seq_len, hidden_dim] ,x_lengths = [batch_size,]
        output, hid = self.decoder(y=y, y_lengths=y_lengths, hid=hid)
        return output, None

    def translate(self, x, x_lengths, y, max_length=10):
        # infer 阶段
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        # 循环逐步预测：设定最大的预测步长 max_length
        for i in range(max_length):
            output, hid = self.decoder(y=y, y_lengths=torch.ones(batch_size).long().to(y.device), hid=hid)
            # output = [batch_size, 1, vocab_size]
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)

        return torch.cat(preds, 1), None


# masked cross entropy loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # input: (batch_size * seq_len) * vocab_size
        input = input.contiguous().view(-1, input.size(2))
        # target: [batch_size, seq_len] ---> [batch_size * seq_len]
        target = target.contiguous().view(-1, 1)
        # mask = [batch_size, seq_len] ---> [batch_size * seq_len]
        mask = mask.contiguous().view(-1, 1)
        #
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


dropout = 0.2
hidden_size = 100
encoder = PlainEncoder(vocab_size=en_total_words,
                       hidden_size=hidden_size,
                       dropout=dropout)
decoder = PlainDecoder(vocab_size=cn_total_words,
                       hidden_size=hidden_size,
                       dropout=dropout)
model = PlainSeq2Seq(encoder, decoder)
model = model.to(device)
loss_fn = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())


def evaluate(model, data):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss / total_num_words)


def train(model, data, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        total_num_words = total_loss = 0.
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            # trg的输入:去除最后的 <EOS> 字符 [batch_size,seq-1]
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            # trg的输出:去除最开始的 <SOS> 字符  [batch_size,seq-1]
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            # 所有的句子都减去 1 个字符长度
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()
            mb_y_len[mb_y_len <= 0] = 1

            # mb_pred = [batch_size,seq_len, vocab_size]
            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            # 使用最大 mb_y_len 的方法的生产 mask，其实可以用pad的方法啦
            # mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            # mb_out_mask = mb_out_mask.float()
            mb_out_mask = (mb_output != 1).float()

            # 计算 loss： mb_output = [batch_size,seq-1] ，
            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            # 更新模型
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            # if it % 100 == 0:
            #     print("Epoch", epoch, "iteration", it, "loss", loss.item())

        print("Epoch", epoch, "Training loss", total_loss / total_num_words)
        if epoch % 5 == 0:
            evaluate(model, dev_data)


# train(model, train_data, num_epochs=20)


# In[58]:


def translate_dev(i):
    en_sent = " ".join([inv_en_dict[w] for w in dev_en[i]])
    print(en_sent)
    cn_sent = " ".join([inv_cn_dict[w] for w in dev_cn[i]])
    print("".join(cn_sent))

    mb_x = torch.from_numpy(np.array(dev_en[i]).reshape(1, -1)).long().to(device)
    mb_x_len = torch.from_numpy(np.array([len(dev_en[i])])).long().to(device)
    bos = torch.Tensor([[cn_dict["BOS"]]]).long().to(device)

    translation, attn = model.translate(mb_x, mb_x_len, bos)
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "EOS":
            trans.append(word)
        else:
            break
    print("".join(trans))


# for i in range(100, 120):
#     translate_dev(i)
#     print()


# 数据全部处理完成，现在我们开始构建seq2seq模型
# #### Encoder
# - Encoder模型的任务是把输入文字传入embedding层和GRU层，转换成一些hidden states作为后续的context vectors

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        # encoder 部分改成双向的 GRU
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        hid = torch.cat([hid[-2], hid[-1]], dim=1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out, hid


# #### Luong Attention
# - 根据context vectors和当前的输出hidden states，计算输出

class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)

    def forward(self, output, context, mask):
        # output: batch_size, output_len, dec_hidden_size
        # context: batch_size, context_len, 2*enc_hidden_size

        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)

        # 做线性映射, 使得 context 的 hidden_size 与 decoder的hidden_size 一致
        # [ batch_size, context_len, encoder_size] ---> [ batch_size, context_len, dec_hidden_size]
        context_in = self.linear_in(context.view(batch_size * input_len, -1)).view(
            batch_size, input_len, -1)  # batch_size, context_len, dec_hidden_size

        # 计算 attention score : query = output, key =  context
        # output =[ batch_size, output_len, dec_hidden_size]
        # context_in.transpose(1,2)= [batch_size, dec_hidden_size, context_len ]
        # atten = [batch_size, output_len, context_len ] ,这儿的 attention 是 输出相对于 context的所有的 attention
        attn = torch.bmm(output, context_in.transpose(1, 2))

        # 对 attention 做 mask = [batch_size，context_len]
        attn.data.masked_fill(mask.unsqueeze_(dim=1)==0, -1e6)

        # 在 dim =2 的维度计算 attention score 的 softmaxt 分布
        # [batch_size, output_len, context_len]
        attn = F.softmax(attn, dim=2)
        # 获取 attention 后的向量 [batch_size, output_len, enc_hidden_size]
        context = torch.bmm(attn, context)

        # 对 context和output 进行叠加
        output = torch.cat((context, output), dim=2)  # batch_size, output_len, hidden_size*2

        # 在进行一个全连接层
        output = output.view(batch_size * output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)
        return output, attn


# #### Decoder
# - decoder会根据已经翻译的句子内容，和context vectors，来决定下一个输出的单词
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self,src, x_len=None, y_len=None):
        # a mask of shape x_len * y_len
        # device = x_len.device
        # # 获取最大值
        # max_x_len = x_len.max()
        # max_y_len = y_len.max()
        # #
        # x_mask = torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:, None]
        # y_mask = torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]
        # mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        mask  = (src !=1)
        return mask

    def forward(self, ctx, ctx_lengths, y, y_lengths, hid, src):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))  # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        # 使用 tokenizer 的 src 来生成 mask
        mask = self.create_mask(src) # y_lengths, ctx_lengths)

        # 进行 attention
        output, attn = self.attention(output_seq, ctx, mask)
        output = F.log_softmax(self.out(output), -1)

        return output, hid, attn


# #### Seq2Seq
# - 最后我们构建Seq2Seq模型把encoder, attention, decoder串到一起

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid, attn = self.decoder(ctx=encoder_out,
                                         ctx_lengths=x_lengths,
                                         y=y,
                                         y_lengths=y_lengths,
                                         hid=hid,
                                         src = x)
        return output, attn

    def translate(self, x, x_lengths, y, max_length=100):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(ctx=encoder_out,
                                             ctx_lengths=x_lengths,
                                             y=y,
                                             y_lengths=torch.ones(batch_size).long().to(y.device),
                                             hid=hid,
                                             src = x)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
            attns.append(attn)
        return torch.cat(preds, 1), torch.cat(attns, 1)


# 训练
dropout = 0.2
embed_size = hidden_size = 100
encoder = Encoder(vocab_size=en_total_words,
                  embed_size=embed_size,
                  enc_hidden_size=hidden_size,
                  dec_hidden_size=hidden_size,
                  dropout=dropout)
decoder = Decoder(vocab_size=cn_total_words,
                  embed_size=embed_size,
                  enc_hidden_size=hidden_size,
                  dec_hidden_size=hidden_size,
                  dropout=dropout)
model = Seq2Seq(encoder, decoder)
model = model.to(device)
loss_fn = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())


train(model, train_data, num_epochs=30)



for i in range(100, 120):
    translate_dev(i)
    print()
