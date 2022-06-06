#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/20 16:35 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/20 16:35   wangfc      1.0         None
"""
import os

from data_process.data_generator import TextClassifierDataGenerator
from data_process.data_processor import DataProcessor
from bert4keras.tokenizers import Tokenizer


class SentimentDataProcessor(DataProcessor):
    """数据生成器
    """

    def __init__(self, data_dir, output_dir, tokenizer=None, vocab_file=None, max_seq_length=512, train_batch_size=32,
                 do_lower_case=True):
        if tokenizer is None:
            tokenizer = self.create_tokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        super().__init__(data_dir=data_dir, output_dir=output_dir, tokenizer=tokenizer, max_seq_length=max_seq_length,
                         train_batch_size=train_batch_size)
        self.data_types = ['train', 'valid', 'test']

    def create_tokenizer(self, vocab_file, do_lower_case=True):
        tokenizer = Tokenizer(vocab_file, do_lower_case=do_lower_case)
        return tokenizer

    def load_data(self, filename):
        """加载数据
        单条格式：(文本, 标签id)
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                text, label = l.strip().split('\t')
                D.append((text, int(label)))
        return D

    def get_examples(self, data_type='train'):
        data_filename = f"sentiment.{data_type}.data"
        data_file = os.path.join(self.data_dir, data_filename)
        examples = self.load_data(filename=data_file)
        return examples

    def create_generator(self, data_type):
        examples = self.get_examples(data_type=data_type)
        g = TextClassifierDataGenerator(data=examples, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        return g

    def prepare_data(self, data_type):
        """
        实现 父类的 prepare_data 方法
        """
        return self.create_generator(data_type=data_type)