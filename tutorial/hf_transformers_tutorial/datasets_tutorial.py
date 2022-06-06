#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/9 14:46 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/9 14:46   wangfc      1.0         None


datasets是huggingface维护的一个轻量级可扩展的数据加载库，其兼容pandas、numpy、pytorch和tensorflow，使用简便。
根据其官方简介：Datasets originated from a fork of the awesome TensorFlow Datasets，
https://github.com/huggingface/datasets

datasets是源自于tf.data的，两者之间的主要区别可参考这里。

tf.data相较于pytorch的dataset/dataloader来说，（个人认为）其最强大的一点是可以处理大数据集，而不用将所有数据加载到内存中。
datasets的序列化基于Apache Arrow（tf.data基于tfrecord），熟悉spark的应该对apache arrow略有了解。
datasets使用的是内存地址映射的方式来直接从磁盘上来读取数据，这使得他能够处理大量的数据。用法简介可参考Quick tour。
下面对datasets用法做一些简单的记录。



"""
import os
import datasets
from datasets import list_metrics,list_datasets,load_dataset
from  pprint import pprint
# datasets提供了许多NLP相关的数据集，使用list_datasets()
dataset_ls =  datasets.list_datasets()

metrics = list_metrics()

print(f"🤩 Currently {len(dataset_ls)} datasets are available on the hub:")
pprint(dataset_ls, compact=True)
print(f"🤩 Currently {len(metrics)} metrics are available on the hub:")
pprint(metrics, compact=True)

os.getcwd()

dataset_name = 'conll2003'
conll2003_dir = os.path.join('home','wangfc','faq','corpus',dataset_name)
conll_data = load_dataset(dataset_name,cache_dir=conll2003_dir)

"""
load数据集的时候会从datasets github库中拉取读取csv数据的脚本，用此脚本来读取本地数据。
但是在读取的过程中非常容易出现网络错误，这里的做法是直接将github 库中的csv读取脚本直接下载到本地datasets安装库中

"""

conll_data = datasets.load_dataset(path="conll2003" ,cache_dir=data_dir)

