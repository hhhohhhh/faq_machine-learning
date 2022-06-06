#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/3/2 14:12 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/2 14:12   wangfc      1.0         None

"""
import os
from typing import List
import numpy as np
"""
Pytorch的数据读取主要包含三个类:

Dataset
DataLoader
DataLoaderIter
这三者大致是一个依次封装的关系: 1.被装进2., 2.被装进3.

class CustomDataset(Dataset):
   # 自定义自己的dataset

dataset = CustomDataset()
dataloader = Dataloader(dataset, ...)

for data in dataloader:
   # training...
在for 循环里, 总共有三点操作:

调用了dataloader 的__iter__() 方法, 产生了一个DataLoaderIter
反复调用DataLoaderIter 的__next__()来得到batch, 具体操作就是, 多次调用dataset的__getitem__()方法 (如果num_worker>0就多线程调用), 然后用collate_fn来把它们打包成batch. 中间还会涉及到shuffle , 以及sample 的方法等, 这里就不多说了.
当数据读完后, __next__()抛出一个StopIteration异常, for循环结束, dataloader 失效.


"""

from torch.utils.data import Dataset
from utils.utils import default_load_json



class BasicDataset(Dataset):
    """
    @author:wangfc
    @desc:
        继承 Dataset 类，只需要自己重构 2个函数：
        __getitem__()
        __len__()
    @version：
    @time:2021/3/3 17:45

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self,features=None,data_type='train'):
        self.features = features
        self.data_type = data_type

    def __getitem__(self, index):
        """
        @author:wangfc
        @desc:  获取 一个 triplet
        @version：
        @time:2021/3/2 14:20

        Parameters
        ----------

        Returns
        -------
        """
        # 获取每个训练样本的feature
        feature = self.features[index]
        # 数据预处理：将feature 中的特征转换为 tensor
        # data = self.data_preproccess(feature=feature)

        return feature

    def __len__(self):
        return len(self.features)



class TripletExample(object):
    """
    @author:wangfc
    @desc: 定义每个 TripletExample 类
    @version：
    @time:2021/3/2 17:30

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self, guid=None,anchor=None,positive=None,negative=None):
        self.guid=guid
        self._anchor = anchor
        self._positive = positive
        self._negative = negative

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self,value):
        self._anchor =value

    @property
    def positive(self):
        return self._positive

    @property
    def negative(self):
        return self._negative



class TripleLetExampleLoader(object):
    """
    @author:wangfc
    @desc:  
    定义如何读取 TripletExample,
    默认 triplet 按照 json格式存储
    @version：
    @time:2021/3/2 17:31

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self,data_path):
        self.data_path =data_path

    def __call__(self, *args, **kwargs)->List[TripletExample]:
        examples= []
        data_json_ls = default_load_json(json_file_path=self.data_path)
        for data_dict in data_json_ls:
            example = TripletExample(**data_dict)
            examples.append(example)
        return examples


class SentenceFeature():
    def __init__(self,tokens,input_ids,token_type_ids,attention_mask,**kwargs):
        self.tokens= tokens
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask

    @property
    def bert_feature(self):
        return dict(input_ids= self.input_ids,token_type_ids=self.token_type_ids,attention_mask=self.attention_mask)

    @property
    def as_numpy_array(self):
        # shape = (3,max_seq)
        arr =  np.stack([self.input_ids,self.token_type_ids,self.attention_mask],axis=0)
        return arr



class TripletFeature():
    """
    @author:wangfc
    @desc: 定义 Triplet 的feature
    @version：
    @time:2021/3/2 17:45

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self,guid:int=None,
                 anchor_feature:SentenceFeature=None,
                 positive_feature:SentenceFeature=None,
                 negative_feature:SentenceFeature=None):
        self.guid=guid
        self.anchor_feature = anchor_feature
        self.positive_feature = positive_feature
        self.negative_feature = negative_feature

    def get_feature_dict(self):
        return dict(anchor_feature=self.anchor_feature.as_numpy_array,
                    positive_feature=self.positive_feature.as_numpy_array,
                    negative_feature=self.negative_feature.as_numpy_array)



class TripletFeatureConvert():
    def __init__(self):
        self.tokenizer = None

    def convert_example_to_feature(self):
        pass






class TripletDataset(BasicDataset):
    """
    @author:wangfc
    @desc:
        继承 Dataset 类，只需要自己重构 2个函数：
        __getitem__()
        __len__()
    @version：
    @time:2021/3/3 17:45

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self,features=None,data_type='train'):
        self.features = features
        self.data_type = data_type

    def __getitem__(self, index):
        """
        @author:wangfc
        @desc:  获取 一个 triplet
        @version：
        @time:2021/3/2 14:20

        Parameters
        ----------

        Returns
        -------
        """
        # 获取每个训练样本的feature
        feature = self.features[index]
        # 数据预处理：将feature 中的特征转换为 tensor
        # data = self.data_preproccess(feature=feature)

        return feature

    def __len__(self):
        return len(self.features)


    def data_preproccess(self, feature:TripletFeature):
        '''
        数据预处理
        :param data:
        :return:
        '''
        anchor_feature = feature.anchor_feature
        data = feature
        return data






