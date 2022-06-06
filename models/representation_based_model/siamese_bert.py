#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2021/3/1 16:34 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/1 16:34   wangfc      1.0         None

"""

from typing import Union, List

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, AlbertModel

from data_process import SentenceLabelExample, SentenceLabelFeature, SentenceLabelDataset
from utils.torch.utils import  set_batch_to_device


from ..model import BasicModel
from evaluation.InformationRetrievalEvaluator import InformationRetrievalEvaluator


import logging
logger = logging.getLogger(__name__)


class SiameseBert(BasicModel):
    def __init__(self,
                 pretrained_bert_dir,keep_dim=True,output_embedding_size=None,
                 max_length=128,
                 tokenizer=None,
                 device =None,
                 pooling_strategy='mean_pooling', # 可以设计不同的 pooling_strategy
                 if_build_triplet_data=False,
                 if_build_sentence_label_data=True,
                 data_pipeline='sentence_label_data', # 针对不同的模型，可以设计不同 data_pipeline

                 ):
                 # output_size,hidden_size,num_layers,bidirectional,dropout):
        super().__init__()

        self.max_length = max_length
        self.tokenizer =  tokenizer # BertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_bert_dir)
        self.device = device
        self.transformer = AlbertModel.from_pretrained(pretrained_model_name_or_path=pretrained_bert_dir)
        self.pooling_strategy = pooling_strategy
        self.if_build_triplet_data = if_build_triplet_data
        self.if_build_sentence_label_data = if_build_sentence_label_data

        # embedding_dim = self.model.config.to_dict()['hidden_size']
        # self.gru =  nn.GRU(input_size=embedding_dim ,hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional,
        #                    batch_first=True,dropout=0 if num_layers<2 else dropout)
        # self.dropout =nn.Dropout(dropout)
        # self.linear = nn.Linear(in_features=hidden_size*2 if bidirectional else hidden_size , out_features=output_size)


    @classmethod
    def tokenizer_fn(self,batch:int,tokenizer:BertTokenizer,max_length:int):
        sentences = batch.get('text')
        labels = batch.get('label')
        # <class 'transformers.tokenization_utils_base.BatchEncoding'>
        sentence_encodings = tokenizer(sentences, padding=True, truncation=True, max_length=max_length,
                                           return_tensors='pt')
        # 转换为 Dict[str:tensor]
        sentence_encodings_dict = {key:value for key,value in sentence_encodings.items()}
        return {"sentence_encodings":sentence_encodings_dict,"labels":labels}


    def mean_pooling(self, tokens_encoder_output, attention_mask):
        """
        @author:wangfc
        @desc:   Mean Pooling - Take attention mask into account for correct averaging
        考虑了 mask，更加合理一些
        相关论文证明 Average Embedding 效果好于 [CLS]
        @version： 来自 https://blog.khay.site/2020/09/06/FAQ%E4%B9%8B%E5%9F%BA%E4%BA%8EBERT%E7%9A%84%E5%90%91%E9%87%8F%E8%AF%AD%E4%B9%89%E6%A3%80%E7%B4%A2%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/

        @time:2021/3/5 9:00

        Parameters
        ----------

        Returns
        -------
        """
        # 获得句子中每个 token 的向量表示
        # token_embeddings = (batch_sizse, max_seq_length,hidden_size )
        # token_embeddings = model_output[0]
        # attention_mask= (batch_size, max_seq_length) -> (batch_size, max_seq_length,1) -> (batch_size, max_seq_length,hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(tokens_encoder_output.size()).float()
        # 对 每个token 进行 mask 并相加
        sum_embeddings = torch.sum(tokens_encoder_output * input_mask_expanded, 1)
        # 计算每句话有效的 mask数量
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # 求平均
        pool_output = sum_embeddings / sum_mask
        return pool_output

    def max_pooling(self):
        pass



    def forward(self,batch):
        if self.if_build_triplet_data:
            output = self.forward_with_triplet_dataset(batch)
        elif self.if_build_sentence_label_data:
            output = self.forward_with_sentence_label_dataset(batch)
        return output



    def forward_with_sentence_label_dataset(self,batch_encoding):
        """
        @author:wangfc
        @desc:
        输入： 每个step训练所需的batch： sentence_label_dataset = (text,label)
        输出： 对应的  embeddings

        @version：
        @time:2021/3/5 16:58

        Parameters
        ----------

        Returns
        -------
        """

        sentence_encoding = batch_encoding.get("sentence_encodings")
        labels = batch_encoding.get("labels")
        # transformer_output :BaseModelOutputWithPooling = odict_keys(['last_hidden_state', 'pooler_output'])
        # last_hidden_state= (batch_size, max_sequence_length, embedding_dim)
        # pooler_output = transformer_output[1]
        transformer_output = self.transformer(**sentence_encoding)
        if not self.pooling_strategy:
            # 直接使用 transformer_output 的 pooler_output 作为句子的表征
            output = transformer_output.get('pooler_output')
        else:
            # last_hidden_state =  (batch_size, max_sequence_length, embedding_dim)
            tokens_encoder_output = transformer_output.get('last_hidden_state')
            # 在Bert中，pool的作用是，输出的时候，用一个全连接层将整个句子的信息用第一个token来表示,考虑 mask？
            attention_mask = sentence_encoding['attention_mask']
            if self.pooling_strategy == 'mean_pooling':
                pool_out = self.mean_pooling(tokens_encoder_output=tokens_encoder_output, attention_mask=attention_mask)
            elif self.pooling_strategy == 'max_pooling':
                pool_out = self.max_pooling(tokens_encoder_output = tokens_encoder_output,attention_mask = attention_mask)
            output = pool_out
        output_dict = {'embeddings':output,'labels':labels}
        return output_dict

    def forward_with_triplet_dataset(self, batch):
        """
        @author:wangfc27441
        @desc:
        输入： 每个step训练所需的batch： triplet = (anchor,positive,negative)
        输出： 对应的 triplet_embedding = (anchor_emb,pos_emb,neg_emb)
        @version：
        @time:2021/3/2 10:06

        Parameters
        ----------
        batch： triplet = (anchor,positive,negative)
                shape = (batch_size,3,max_sequence_length)
        Returns
        -------
                triplet_embedding =(anchor_emb,pos_emb,neg_emb)
                shape = (batch_size,3,embedding_size)

        """
        batch_size = batch.get('guid').__len__()
        # 对 anchor, positive,negative 进行 tokenize
        input_ids_ls = []
        sentences = []

        for role in ['anchor','positive','negative']:
            role_sentences = batch.get(role)
            # 将 'anchor','positive','negative' 放在同一个 list当中
            sentences.extend(role_sentences)
            # 分别记录 'anchor','positive','negative' 的 input_ids
            role_sentences_input_ids = self.tokenizer(role_sentences, padding=True,truncation=True, max_length=self.max_length, return_tensors='pt')
            input_ids_ls.append(role_sentences_input_ids)

        batch_encoding = self.tokenizer(sentences,padding=True,truncation=True, max_length=self.max_length, return_tensors='pt')

        # transformer_output = (sequence_output,pool_output)
        transformer_output = self.transformer(**batch_encoding)
        #  cnn 或者 bert model 的时候： batch first
        # text = [batch_size,len_sentence]
        # sequence_output = [batch_size, max_sequence_length, embedding_dim]
        # pool_output = [batch_size,  embedding_dim]

        if not self.if_pooling:
            # cls_output= (batch_size, max_sequence_length, embedding_dim)
            output = transformer_output[1]
        else:
            # sequence_output =  (batch_size, max_sequence_length, embedding_dim)
            sequence_output = transformer_output[0]
            (totoal_batch_size, max_sequence_length, embedding_dim) = sequence_output.shape
            # 在Bert中，pool的作用是，输出的时候，用一个全连接层将整个句子的信息用第一个token来表示
            # 不考虑mask，直接使用 pool 层：
            # pool_out = F.avg_pool2d(input= sequence_output,kernel_size=(max_sequence_length,1)).squeeze(1)
            # 考虑 mask？
            attention_mask = batch_encoding['attention_mask']
            pool_out = self.mean_pooling(model_output=transformer_output,attention_mask=attention_mask)
            output = pool_out

        # 如何进行 triplet的训练
        # batch： triplet = (anchor,positive,negative)
        #         shape = (batch_size,3,max_sequence_length)
        # input_ids_reshaped = (3,batch_size,max_seq)

        _, embedding_size = output.shape

        output_reshaped = output.reshape(3,batch_size,embedding_size)

        # input_ids_stacked = (batch_size,3,max_seq)
        output_stacked = torch.stack(list(output_reshaped),dim=1)
        # logger.info(f"output.shape={output.shape},input_ids_stacked.shape={output_stacked.shape}")
        return {'triplet_embedding':output_stacked}


    def convert_sentences_to_dataset(self,sentences:List[str],data_type='test')->SentenceLabelDataset:
        """
        将句子直接转换为 dataset
        sentence = {id:text}
        """
        features = []
        for guid,text in enumerate(sentences):
            example = SentenceLabelExample(guid=guid,text=text)
            feature = SentenceLabelFeature(guid=example.guid,id=example.id, text=example.text,label=example.label)
            features.append(feature)
            # if guid %10000==0:
            #     logger.info(f"转换第{guid}个example到feature")
        dataset = SentenceLabelDataset(examples=features, data_type=data_type)
        logger.info(f"sentences={len(sentences)}转换dataset={dataset}")
        return dataset


    def encode(self,
               sentences: Union[str, List[str], List[int]]= None,
               dataloader: DataLoader = None,
               batch_size: int = 32,
               show_progress_bar: bool = True,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               is_pretokenized: bool = False,
               # device: str = None,
               num_workers: int = 0) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param is_pretokenized: If is_pretokenized=True, sentences must be a list of integers, containing the tokenized sentences with each token convert to the respective int.
        :param device: Which torch.device to use for the computation
        :param num_workers: Number of background-workers to tokenize data. Set to positive number to increase tokenization speed
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel()==logging.INFO or logging.getLogger().getEffectiveLevel()==logging.DEBUG)

        input_was_string = False

        self.to(self.device)
        all_embeddings = []

        if dataloader is None and sentences is not None:
            # 直接输入句子，转换为 dataloader
            if isinstance(sentences, str):  # Cast an individual sentence to a list with length 1
                sentences = [sentences]
                input_was_string = True

            # # length_sorted_idx: 从小到大排序后，第 i 个 位置的 index,表示 原来的数据在排序后的序列中的位置
            length_sorted_idx = np.argsort([len(text) for text in sentences])
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
            # # 将原始的 sentence 转换为 dataset
            # inp_dataset = EncodeDataset(sentences_sorted, model=self, is_tokenized=is_pretokenized)
            # 生成 dataloader
            # inp_dataloader = DataLoader(inp_dataset, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)
            inp_dataset = self.convert_sentences_to_dataset(sentences=sentences_sorted)
            inp_dataloader = DataLoader(inp_dataset, batch_size=batch_size, num_workers=num_workers,
                                        shuffle=False)

        elif dataloader is not None:
            inp_dataloader = dataloader
        else:
            raise ValueError("没有数据可以进行 evaluate")

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Evaluating batches")

        for batch in iterator:
            # 对 sentences 进行 tokenize
            batch_encoding = self.tokenizer_fn(batch,tokenizer=self.tokenizer,max_length=self.max_length)
            # 将 batch 放置在 device
            batch_encoding_on_device = set_batch_to_device(batch= batch_encoding,device=self.device)

            with torch.no_grad():
                # 输入模型
                out_features = self.forward(batch_encoding_on_device)
                embeddings = out_features["embeddings"]

                if output_value == 'token_embeddings':
                    #Set token embeddings to 0 for padding tokens
                    input_mask = out_features['attention_mask']
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                for emb in embeddings:
                    all_embeddings.append(emb.detach())
        # 恢复原来的排列顺序
        # all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = [all_embeddings[idx] for idx in length_sorted_idx]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.cpu().numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


    def evaluator(self,queries,corpus,relevant_docs):
        """
        @author:wangfc
        @desc:
        使用 InformationRetrievalEvaluator 作为 evaluator

        @version：
        @time:2021/3/18 22:27

        Parameters
        ----------
            queries： id：sentence的字典
            corpus： id：sentence 的 字典
            relevant_docs： id：相关句子的id的字典
        Returns
        -------
        """
        return InformationRetrievalEvaluator(queries=queries,corpus=corpus,relevant_docs=relevant_docs)


class SiameseNet(BasicModel):
    pass
