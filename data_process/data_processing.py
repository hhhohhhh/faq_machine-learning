#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/1/7 16:32

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/7 16:32   wangfc      1.0         None


"""
import os
import re
from math import ceil
from typing import List, Text, Union, Dict, Tuple
import pandas as pd
import jieba
from collections import Counter
import logging

from sklearn.model_selection import train_test_split

from tokenizations.tokenization import is_not_punctuation_or_whitespace
from utils.io import dataframe_to_file

logger = logging.getLogger(__name__)


def get_tokens_counter(data, column='sentence', stopwords=None):
    data_tokens = data.loc[:, column].apply(lambda text: jieba_tokenize(text, stopwords=stopwords))
    # 进行分词
    tokens_counter = Counter()
    for index in range(data_tokens.__len__()):
        tokens = data_tokens.iloc[index]
        tokens_counter.update(tokens)
    return tokens_counter


def jieba_tokenize(text, stopwords=None) -> List[Text]:
    tokens = jieba.lcut(text)
    if stopwords is not None:
        tokens = list(filter(lambda x: x not in stopwords, tokens))
    return tokens


def percent_in_vocab(sentence: Text, vocab=None, percent=0.6):
    """
    @time:  2021/7/28 15:33
    @author:wangfc
    @version:
    @description: 检查 sentence 中存在 vocab 中的比例，如果不在 vocab 中的比例过高 大于 percent，进行过滤

    @params:
    @return:
    """
    in_vocab_char = 0
    for x in sentence:
        if x.lower() in vocab:
            # print(x)
            in_vocab_char += 1
    in_vocab_percent = in_vocab_char / sentence.__len__()
    if in_vocab_percent > percent:
        return True
    else:
        return False


def generate_n_gram(x: List[str], n: int = 2):
    n_grams = set((zip(*[x[i:] for i in range(n)])))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


def is_security_code(text, security_code_pattern=r'^\d{6}$', only_digit_pattern=False, digit_pattern=r'^\d+$'):
    """
     # 非6位的纯数字，都是faq, 六位的除股票外，还有可能基金债券代码，客户要求也按advice来
    """
    if only_digit_pattern:
        re_compiled = re.compile(digit_pattern)
    else:
        re_compiled = re.compile(security_code_pattern)

    if re_compiled.match(text):
        return True
    else:
        return False


def remove_punctuation(text: Text):
    assert isinstance(text, str)
    text = "".join(list(filter(is_not_punctuation_or_whitespace, text)))
    return text


def remove_line_break_char(text:Text):
    if isinstance(text,str):
        text = text.replace('\n\r','').replace('\r\n','').replace('\n','')
    return text


class DataSpliter():
    def __init__(self, data: pd.DataFrame, output_dir, train_size=0.8, dev_size=0, test_size=0.1,
                 train_filename='raw_train.json', dev_filename='raw_dev.json', test_filename='raw_test.json',
                 split_strategy=None, stratify_column=None, if_filter_single_label_test_data=True):
        self._data = data
        self._output_dir = output_dir
        self._train_size = train_size
        self._dev_size = dev_size
        self._test_size = test_size
        self._train_filename = train_filename
        self._dev_filename = dev_filename
        self._test_filename = test_filename
        # 分割数据的策略
        self._split_strategy = split_strategy
        self._stratify_column = stratify_column
        # 是否过滤单标签的测试数据
        self._if_filter_single_label_test_data = if_filter_single_label_test_data

        self._train_data_path = os.path.join(output_dir, self._train_filename)
        self._dev_data_path = os.path.join(output_dir, self._dev_filename)
        self._test_data_path = os.path.join(output_dir, self._test_filename)

    def get_split_data(self):
        if os.path.exists(self._train_data_path) and os.path.exists(self._test_data_path):
            train_data, dev_data, test_data = self._load_split_data()
        else:
            train_data, dev_data, test_data = self._split_data()
        return train_data, dev_data, test_data


    def _split_data(self, split_strategy=None):
        if split_strategy is None:
            split_strategy = self._split_strategy

        dev_in_left_size = (1 - self._train_size - self._test_size) / (1 - self._train_size)
        if split_strategy is None:
            # 随机分割数据
            train_data, dev_and_test_data = train_test_split(self._data, train_size=self._train_size)
        elif split_strategy == 'stratify':
            logger.info('{}\n采用分层采样的方式，先分割原始数据集，再生产example'.format('*' * 100))
            # 获取 label（可能存在多标签的情况） 列对应的index 作为 stratify_index
            label_to_index_dict, label_value_count = self._create_label_to_index(data=self._data)
            self._label_to_index_dict = label_to_index_dict
            stratify_index = self._build_stratify_index(data=self._data, label_to_index_dict=self._label_to_index_dict)
            # 某些数量比较少的数据， 使用 stratify 分层抽样因为数据过少会报错
            train_data, dev_and_test_data = self._train_and_split(data=self._data,
                                                                  train_size=self._train_size,
                                                                  stratify=stratify_index,
                                                                  label_value_count=label_value_count,
                                                                  )

        if self._dev_size == 0:
            dev_data = None
            test_data = dev_and_test_data
        elif split_strategy is None:
            dev_data, test_data = train_test_split(dev_and_test_data, train_size=dev_in_left_size)
        elif split_strategy == 'stratify':
            _, label_value_count = self._create_label_to_index(data=dev_and_test_data)
            stratify_index = self._build_stratify_index(data=dev_and_test_data,
                                                        label_to_index_dict=self._label_to_index_dict)
            dev_data, test_data = self._train_test_split(dev_and_test_data, train_size=dev_in_left_size,
                                                   stratify=stratify_index,label_value_count = label_value_count)

        if self._if_filter_single_label_test_data:
            test_data = self._filter_single_label_test_data(test_data=test_data)

        self._save_split_data(train_data, dev_data, test_data)
        return train_data, dev_data, test_data

    def _build_stratify_index(self, data, label_to_index_dict)->pd.Series:
        stratify_index = data.loc[:,self._stratify_column].apply(
            lambda label: self._get_label_index(label, label_to_index_dict))
        return stratify_index

    def _get_label_index(self, label: Union[Text, List[Text]], label_to_index_dict: Dict[Text, int]):
        """"""
        if isinstance(label, str):
            index = label_to_index_dict[label]
        elif isinstance(label, list):
            labels_as_string = self._labels_as_string(labels=label)
            index = label_to_index_dict[labels_as_string]
        else:
            raise TypeError
        return index

    def _labels_as_string(self, labels=List[Text]):
        return "_".join(sorted(labels))

    def _create_label_to_index(self, data) -> Tuple[Dict[Text, int], pd.Series]:
        """
        创建 label （可能存在多标签的情况） 列对应的index 的映射关系
        """
        if self._stratify_column is None:
            raise ValueError(f"当使用 split_strategy ={self._split_strategy} 的时候，_stratify_column 不能为空")

        labels = data.loc[:, self._stratify_column].apply(self._convert_label_to_string)
        label_value_count = labels.value_counts()
        label_to_index = {label: index for index, label in enumerate(label_value_count.index)}
        return label_to_index, label_value_count

    def _convert_label_to_string(self, label):
        if isinstance(label, str):
            labels_as_string = label
        elif isinstance(label, list):
            labels_as_string = self._labels_as_string(labels=label)
        else:
            raise TypeError
        return labels_as_string

    def _train_and_split(self, data, train_size, stratify=None, label_value_count=None, **options):
        # 某些数量比较少的数据， 使用 stratify 分层抽样因为数据过少会报错
        # 计算最少所需的 样本数量
        minimum_size = self._minimum_size(train_size)
        # 过滤不满足最少样本数量的 labels
        less_size_labels = label_value_count[label_value_count<minimum_size].index.tolist()
        # 判断是否属于这些不满足最少样本数量的 labels
        is_in_less_size_labels = data.loc[:,self._stratify_column].apply(lambda label: self._convert_label_to_string(label) in less_size_labels)
        # 过滤 这些不满足最少样本数量的 labels的数据
        less_size_data = data.loc[is_in_less_size_labels].copy()
        # 过滤 满足最少样本数量的 labels的数据
        enough_size_data = data.loc[~is_in_less_size_labels].copy()
        # 获取对应的 stratify_index
        enough_size_stratify_index = stratify.loc[enough_size_data.index]
        # 对满足数量的数据进行分层抽样
        train_data,test_data = train_test_split(enough_size_data, train_size=train_size, stratify=enough_size_stratify_index, **options)
        # 合并分割的训练数据和 不满足最少样本数量的 labels的数据
        train_data = pd.concat([train_data,less_size_data])
        return train_data,test_data


    def _minimum_size(self,train_size):
        return max(ceil(1/train_size),ceil(1/(1- train_size)))

    def _filter_single_label_test_data(self, test_data):
        # TODO
        if_single_label = test_data.loc[:,self._stratify_column].apply(is_single_label)
        filter_single_label_test_data =  test_data.loc[if_single_label].copy()
        return  filter_single_label_test_data

    def _save_split_data(self, train_data, dev_data, test_data):
        if dev_data is not None:
            dev_size = len(dev_data)
        else:
            dev_size = 0
        logger.info('split the data_df of {} with train_size={} test_size={} into {} train_df, {} dev_df,'
                    '{} test_df.'.format(len(self._data), self._train_size, self._test_size, len(train_data), dev_size,
                                         len(test_data)))
        for path, data in zip([self._train_data_path, self._dev_data_path, self._test_data_path],
                              [train_data, dev_data, test_data]):
            if data is not None:
                dataframe_to_file(data=data, path=path)


    def _load_split_data(self):
        data_ls = []
        for path in [self._train_data_path, self._dev_data_path, self._test_data_path]:
            data = dataframe_to_file(path=path,mode="r")
            data_ls.append(data)
        return data_ls




def is_single_label(label):
    is_single_label =False
    if isinstance(label,str):
        is_single_label =  True
    elif isinstance(label,list):
        if len(label)==1:
            is_single_label = True
    else:
        raise TypeError
    return is_single_label

if __name__ == '__main__':
    # 蚂蚁金服的数据集
    data_dir = 'data'
    dataset1 = 'atec'
    data_filename1 = 'atec_nlp_sim_train.csv'
    data_filename2 = 'atec_nlp_sim_train_add.csv'
    data_filenames1 = [data_filename1, data_filename2]

    dataset2 = 'ccks'
    data_filename1 = 'task3_train.txt'
    data_filename2 = 'task3_dev.txt'
    data_filenames2 = [data_filename1]

    data_df = get_sentence_pair_data(data_dir=data_dir, dataset=dataset1, data_filenames=data_filenames1)
    # triplet_data_df = build_triplet_data(data_df)
