#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/7 9:28 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/7 9:28   wangfc      1.0         None
"""
from typing import List

from data_process.data_generator import IntentClassifierDataGenerator
from data_process.data_example import InputExample
from data_process.dataset.text_classifier_dataset import TextClassifierProcessor
from CLUE.baselines.models.classifier_utils import TnewsProcessor


class CLUEClassifierProcessor(TextClassifierProcessor):
    """
    该类继承 IntentClassifierProcessor
    重写 get_train_examples() 函数
    重写  create_generator() 函数创建 data generator
    """

    def __init__(self,
                 corpus='corpus',
                 dataset='CLUE',
                 output_data_subdir='tnews',
                 train_data_filename='train.json',
                 dev_data_filename='dev.json',
                 test_data_filename='test.json',
                 output_dir='output/clue_classification_01',
                 label_column='label_desc',
                 text_column='question',
                 raw_data_format='json',
                 stopwords_path=None,
                 vocab_file=None, support_labels=None,
                 max_seq_length=512,
                 train_batch_size=32, test_batch_size=32,
                 transform_train_data_to_rasa_nlu_data=True,
                 use_retrieve_intent=True,
                 use_intent_column_as_label=True
                 ):
        super(CLUEClassifierProcessor, self).__init__(corpus, dataset, output_data_subdir,
                                                      train_data_filename=train_data_filename,
                                                      dev_data_filename=dev_data_filename,
                                                      test_data_filename=test_data_filename,
                                                      output_dir=output_dir,
                                                      vocab_file=vocab_file,
                                                      max_seq_length=max_seq_length,
                                                      train_batch_size=train_batch_size)
        # 获取训练的 labels
        if self.output_data_subdir =='tnews':
            self.labels = TnewsProcessor().get_labels()
            self.support_labels_dict = {label:index for index,label in enumerate(self.labels)}
            self.class_num = self.labels.__len__()


    def get_train_examples(self) -> List[InputExample]:
        """
        重写 get_train_examples() 函数
        """
        if self.output_data_subdir == 'tnews':
            examples = TnewsProcessor(self.support_labels_dict).get_train_examples(data_dir=self.data_dir)
        return examples

    def get_dev_examples(self) -> List[InputExample]:
        if self.output_data_subdir == 'tnews':
            examples = TnewsProcessor(self.support_labels_dict).get_dev_examples(data_dir=self.data_dir)
        return examples

    def get_test_examples(self) -> List[InputExample]:
        if self.output_data_subdir == 'tnews':
            examples = TnewsProcessor(self.support_labels_dict).get_test_examples(data_dir=self.data_dir)
        return examples

    def create_generator(self, data_type, data, batch_size) -> IntentClassifierDataGenerator:
        # examples = self.get_examples(data_type=data_type)
        g = IntentClassifierDataGenerator(data_type=data_type,
                                          data=data, tokenizer=self.tokenizer,
                                          max_seq_length=self.max_seq_length,
                                          batch_size=batch_size
                                          )
        return g