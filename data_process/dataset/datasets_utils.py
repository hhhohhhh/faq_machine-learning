#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/10 9:37 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/10 9:37   wangfc      1.0         None
"""

import os
import datasets
import tensorflow as tf

from utils.io import dump_obj_as_json_to_file,read_json_file
from utils.time import timeit
import logging
logger = logging.getLogger(__name__)

@timeit
def get_conll_data():
    """
    https://www.clips.uantwerpen.be/conll2003/ner/

    数据集第一例是单词，第二列是词性，第三列是语法快，第四列是实体标签。
    在NER任务中，只关心第一列和第四列。实体类别标注采用BIO标注法，
    """
    cwd = os.getcwd()
    script_path = os.path.join('data_process', 'dataset', 'conll2003.py')
    data_dir = os.path.join(cwd, 'corpus', 'conll2003')
    """
    从hugs Face GitHub repo或AWS桶中下载并导入SQuAD python处理脚本(如果它还没有存储在库中)。
    运行SQuAD脚本下载数据集。处理和缓存的SQuAD在一个Arrow 表。
    基于用户要求的分割返回一个数据集。默认情况下，它返回整个数据集。
    """
    conll_data = datasets.load_dataset(path="conll2003", cache_dir=data_dir)

    # Get a sample of train data and print it out:
    for item in conll_data["train"]:
        sample_tokens = item['tokens']
        sample_tag_ids = item["ner_tags"]
        print(sample_tokens)
        print(sample_tag_ids)
        break

    raw_tags_filename = os.path.join('corpus', 'conll2003', 'raw_tags.json')
    if os.path.exists(raw_tags_filename):
        raw_tags = read_json_file(json_path=raw_tags_filename)
    else:
        # The dataset also give the information about the mapping of NER tags and ids:
        dataset_builder = datasets.load_dataset_builder('conll2003', cache_dir=data_dir)
        raw_tags = dataset_builder.info.features['ner_tags'].feature.names
        print(raw_tags)
        dump_obj_as_json_to_file(filename=raw_tags_filename, obj=raw_tags)

    sample_tags = [raw_tags[i] for i in sample_tag_ids]
    print(list(zip(sample_tokens, sample_tags)))
    # Add a special tag <PAD> to the tag set, which is used to represent padding in the sequence.
    tags = ['<PAD>'] + raw_tags
    print(tags)
    return conll_data, tags



