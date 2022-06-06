#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/20 16:47 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/20 16:47   wangfc      1.0         None
"""

import typing
from typing import List, Dict, Text

import os

from data_process.data_analysis import TextClassifierDataAnalysis
from data_process.data_example import InputExample
from data_process.data_feature import InputFeatures, convert_single_example
from data_process.data_generator import TextClassifierDataGenerator, KerasTextClassifierDataGenerator
from data_process.data_processor import DataProcessor, RawDataProcessor
from utils.constants import TRAIN_DATATYPE, EVAL_DATATYPE, TEST_DATATYPE
import logging

from utils.io import read_json_file, dump_obj_as_json_to_file

logger = logging.getLogger(__name__)


class TextClassifierProcessor(DataProcessor):
    """
    TextClassifierProcessor： 主要负责 tf 模型训练的时候提供数据
    """

    def __init__(self,
                 total_support_types_dict=None,
                 total_support_types_dict_name='support_types_dict.json',
                 output_model_dir=None,
                 model_support_types_dict_name="model_support_types_dict.json",
                 tokenizer=None, max_seq_length=None,
                 use_spm=False, do_lower_case=True, multiple=1,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.total_support_types_dict_name = total_support_types_dict_name
        self.total_support_types_dict_path = os.path.join(self.data_dir, self.total_support_types_dict_name)
        if total_support_types_dict is None and os.path.exists(self.total_support_types_dict_path):
            total_support_types_dict = read_json_file(json_path=self.total_support_types_dict_path)
        self.total_support_types_dict = total_support_types_dict
        #
        # self.support_event_types_set = set(self.support_types_dict.keys())
        # self.class_num = self.support_event_types_set.__len__()



        # 动态生成的 support_types_dict
        self.output_model_dir = output_model_dir
        self.model_support_types_dict_name = model_support_types_dict_name
        self.model_support_types_dict_path = os.path.join(self.output_model_dir, self.model_support_types_dict_name)
        self.model_support_types_dict: Dict[Text, Dict[Text, int]] = None


        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_spm = use_spm
        self.do_lower_case = do_lower_case
        self.multiple = multiple

        # if tokenizer is None :
        #     # 当 tokenizer 为空的时候，我们使用 bert/ablert 的方法来生成 tokenizer
        #     self.vocab_file = vocab_file
        #     self.spm_model_file = spm_model_file
        #     tokenizer = FullTokenizer.from_scratch(vocab_file=vocab_file, do_lower_case=do_lower_case,
        #                                            spm_model_file=spm_model_file)
        #     self.vocab = set([k for k, v in tokenizer.vocab.items()])

        self.tokenizer = tokenizer
        # 可以用于 batch_strategy 和 计算 class_weights
        self.train_data_analysis: TextClassifierDataAnalysis = None

        # self.name_to_features = {
        #     "input_ids": tf.io.FixedLenFeature([max_seq_length * multiple], tf.int64),
        #     "input_mask": tf.io.FixedLenFeature([max_seq_length * multiple], tf.int64),
        #     "segment_ids": tf.io.FixedLenFeature([max_seq_length * multiple], tf.int64),
        #     'label_ids': tf.io.FixedLenFeature([class_num], tf.int64),
        #     "is_real_example": tf.io.FixedLenFeature([], tf.int64),
        # }

    def prepare_model_data(self, data_type) -> KerasTextClassifierDataGenerator:
        # 读取 examples
        examples = self._get_examples(data_type=data_type)

        batch_size = self._get_batch_size(data_type=data_type)

        data_analysis = TextClassifierDataAnalysis(data_type=data_type, data_examples=examples)

        self._set_model_support_types_dict(data_type,label_counter = data_analysis.label_counter)

        data_generator = self._create_generator(data_type, examples, batch_size,
                                                data_analysis=data_analysis)

        return data_generator

    def _set_model_support_types_dict(self, data_type: Text, label_counter:typing.Counter):
        """
        从训练数据中 动态获取 support_types_dict
        """
        if (hasattr(self, "model_support_types_dict") ==False) or self.model_support_types_dict is None:
            if data_type != TRAIN_DATATYPE and os.path.exists(self.model_support_types_dict_path):
                model_support_types_dict = read_json_file(json_path=self.model_support_types_dict_path)
            elif data_type == TRAIN_DATATYPE:
                # 从训练数据中 动态获取 support_types_dict
                model_support_types_dict = RawDataProcessor.create_support_types_dict_from_counter(
                    label_counter=label_counter)
                dump_obj_as_json_to_file(obj=model_support_types_dict,filename=self.model_support_types_dict_path)
                logger.info(f"原有 total_support_types_dict 共 {self.total_support_types_dict.__len__()} 个,"
                            f"当前模型共支持 {model_support_types_dict.__len__()} 个")
            else:
                raise ValueError(f"model_support_types_dict不存在")
            self.model_support_types_dict = model_support_types_dict



    def _create_generator(self, data_type, data, batch_size,
                          data_analysis:TextClassifierDataAnalysis) \
            -> TextClassifierDataGenerator:
        data_generator = None
        if data is not None:
            """
            方式1： 
            # 转换为 features
            features = self._convert_examples_to_features(examples=examples)
            # 转换为 dataset 或 generator
            data_generator = self.create_generator(data_type, features, batch_size)
            """
            # 方式2： sample + convert feature + generator
            data_generator = TextClassifierDataGenerator(data_type=data_type,
                                                         data=data, tokenizer=self.tokenizer,
                                                         max_seq_length=self.max_seq_length,
                                                         batch_size=batch_size
                                                         )
        return data_generator

    def _get_batch_size(self, data_type):
        if data_type == TRAIN_DATATYPE:
            batch_size = self.train_batch_size
        elif data_type == EVAL_DATATYPE:
            batch_size = self.eval_batch_size
        elif data_type == TEST_DATATYPE:
            batch_size = self.test_batch_size
        return batch_size

    def _get_examples(self, data_type='train', output_dataframe=False) -> List[InputExample]:
        """See base class."""
        if data_type == TRAIN_DATATYPE:
            examples = self._get_train_examples()
        elif data_type == EVAL_DATATYPE:
            examples = self._get_dev_examples()
        elif data_type == TEST_DATATYPE:
            examples = self._get_test_examples()
        else:
            raise ValueError(f"data_type={data_type} is not exist")

        return examples

    def _convert_examples_to_features(self,
                                      examples: InputExample) -> List[InputFeatures]:
        """Convert a set of `InputExample`s to a TFRecord file."""
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Converting example %d of %d to feature" % (ex_index, len(examples)))

            feature = convert_single_example(ex_index, example,
                                             max_seq_length=self.max_seq_length,
                                             tokenizer=self.tokenizer)
            features.append(feature)
        return features

    def get_labels(self):
        """See base class."""
        return list(self.support_types_dict.keys())
