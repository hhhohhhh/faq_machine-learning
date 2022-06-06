#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/6/8 17:33 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/8 17:33   wangfc      1.0         None
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import csv
import os
import codecs
from pathlib import Path
from typing import Dict, Text, List

from sklearn.model_selection import train_test_split
from data_process.data_example import InputExample
from tokenizations import tokenization
import tensorflow as tf
import pandas as pd
from utils.constants import ALL_DATATYPE, RAW_DATATYPE, TRAIN_DATATYPE, TEST_DATATYPE, EVAL_DATATYPE, \
    RANDOM_BATCH_STRATEGY
from utils.io import read_json_file, dump_obj_as_json_to_file, dataframe_to_file, load_stopwords

logger = logging.getLogger(__name__)


SUPPORT_TYPE_INDEX = "index"
SUPPORT_TYPE_COUNT = 'count'


class RawDataProcessor():
    def __init__(self, corpus,
                 dataset_dir,
                 last_output_subdir=None,
                 output_data_subdir=None,
                 stopwords_path=None,
                 support_labels=None,
                 sample_size_for_label=1000,
                 random_state=None,
                 id_column='id',
                 label_column='intent',
                 text_column='question',
                 raw_data_format='csv',
                 raw_data_filename='raw_data',
                 all_data_filename='all_data',
                 train_data_filename='train',
                 dev_data_filename='dev',
                 test_data_filename='test',
                 new_added_data_subdir=None,
                 new_added_data_filename=None,
                 support_types_dict_filename='support_types_dict.json',
                 min_threshold=100,
                 max_threshold=1000,
                 train_size=0.8,
                 dev_size=0,
                 test_size=0.2
                 ):

        self.corpus = corpus
        self.dataset_dir = dataset_dir

        # 上次版本的数据目录
        self.last_output_subdir = last_output_subdir
        self.last_output_dir = os.path.join(dataset_dir,
                                            last_output_subdir) if self.last_output_subdir is not None else None

        # 新版本的数据目录
        self.output_data_subdir = output_data_subdir
        self.output_data_dir = os.path.join(dataset_dir, output_data_subdir)

        self.stopwords_path = stopwords_path
        self.stopwords = load_stopwords(path=self.stopwords_path)
        # support_labels: 支持训练的类型
        self.support_labels = support_labels

        self.sample_size_for_label = sample_size_for_label
        self.random_state = random_state

        # 各个数据集的名称
        self.id_column = id_column
        self.label_column = label_column
        self.text_column = text_column

        self.raw_data_format = raw_data_format
        self.raw_data_filename = f"{raw_data_filename}.{raw_data_format}"
        self.all_data_filename = f"{all_data_filename}.{raw_data_format}"
        self.train_data_filename = f"{train_data_filename}.{raw_data_format}"
        self.dev_data_filename = f"{dev_data_filename}.{raw_data_format}"
        self.test_data_filename = f"{test_data_filename}.{raw_data_format}"

        # 新增数据
        self.new_added_data_subdir = new_added_data_subdir
        self.new_added_data_filename = new_added_data_filename
        self.new_added_data_path = os.path.join(self.dataset_dir, self.new_added_data_subdir,
                                                self.new_added_data_filename)

        # 分割数据
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.train_size = train_size
        self.dev_size = dev_size
        self.test_size = test_size

        self.support_types_dict_filename = support_types_dict_filename
        self.support_types_dict_json_path = os.path.join(self.output_data_dir, support_types_dict_filename)

    def load_support_types_dict(self):
        if os.path.exists(self.support_types_dict_json_path):
            self.support_types_dict = read_json_file(json_path=self.support_types_dict_json_path)
            self.support_event_types_set = set(self.support_types_dict.keys())
            self.class_num = self.support_event_types_set.__len__()
        else:
            logger.warning(f"support_types_dict_json_path= {self.support_types_dict_json_path}不存在")



    def read_raw_data(self, path: Path = None):
        raise NotImplementedError

    def DataAnalysis(self):
        pass

    def modify_data_label(self) -> pd.DataFrame:
        """
        修改人工修改的数据
        """
        pass

    def get_new_version_data(self):
        self.new_version_data_dict = self.read_data(output_dir=self.output_data_dir)
        raw_data = self.new_version_data_dict.get(RAW_DATATYPE)
        all_data = self.new_version_data_dict.get(ALL_DATATYPE)
        train_data = self.new_version_data_dict.get(TRAIN_DATATYPE)
        dev_data = self.new_version_data_dict.get(EVAL_DATATYPE)
        test_data = self.new_version_data_dict.get(TEST_DATATYPE)

        if self.new_version_data_dict == {}:
            # 读取并且转换 更新的数据
            self.new_added_data_df = self.read_raw_data(path=self.new_added_data_path)
            self.new_add_data_dict = self.train_dev_test_split(raw_data=self.new_added_data_df,
                                                               train_size=self.train_size,
                                                               dev_size=self.dev_size, test_size=self.test_size)
            # 读取最近的数据: all_data, train,dev,test
            self.last_version_data_dict = self.read_data(output_dir=self.last_output_dir)

            # 合并并且分割数据
            self.new_version_data_dict = self.combine_data(new_add_data_dict=self.new_add_data_dict,
                                                           last_version_data_dict=self.last_version_data_dict)
        elif train_data is not None and test_data is not None:
            self.load_support_types_dict()
        elif all_data is not None:
            # 对 数据的修正进行变换
            all_data = self.modify_data_label(data=all_data)
            logger.info(f"修正后的数据共{all_data.shape},分别是:\n{all_data.loc[:, self.label_column].value_counts()}")

            # 对 全部数据进行分割
            new_version_data_dict = self.train_dev_test_split(raw_data=all_data, train_size=self.train_size,
                                                              dev_size=self.dev_size, test_size=self.test_size)
            # 获取支持的类别
            self.new_version_data_dict = new_version_data_dict
            self.support_types_dict = self.get_support_types_dict(new_version_data_dict)
            # 存储数据
            self.new_version_data_dict = self.read_data(mode='w', output_dir=self.output_data_dir,
                                                        all_data=new_version_data_dict.get(ALL_DATATYPE),
                                                        train_data=new_version_data_dict.get(TRAIN_DATATYPE),
                                                        dev_data=new_version_data_dict.get(EVAL_DATATYPE),
                                                        test_data=new_version_data_dict.get(TEST_DATATYPE))
        else:
            raise IOError

        return self.new_version_data_dict

    def read_data(self, output_dir, mode='r',
                  raw_data: pd.DataFrame = None,
                  all_data: pd.DataFrame = None, train_data: pd.DataFrame = None,
                  dev_data: pd.DataFrame = None, test_data: pd.DataFrame = None, **kwargs) -> Dict[Text, pd.DataFrame]:
        """
        @time:  2021/6/8 11:11
        @author:wangfc
        @version:
        @description: 读取最近发版的数据 :'all_data','train','dev','test'

        @params:
        @return:
        """
        data_dict = {}
        data_statistics = {}
        for data_type, data_filename, data in zip([RAW_DATATYPE,
                                                   ALL_DATATYPE, TRAIN_DATATYPE,
                                                   EVAL_DATATYPE, TEST_DATATYPE],
                                                  [self.raw_data_filename,
                                                   self.all_data_filename, self.train_data_filename,
                                                   self.dev_data_filename, self.test_data_filename],
                                                  [raw_data, all_data, train_data, dev_data, test_data]):
            data_path = os.path.join(output_dir, data_filename) if output_dir is not None else None

            if mode == 'r' and data_path and os.path.exists(data_path):
                # data = pd.read_json(path_or_buf=data_path,orient='records',dtype=str)
                # data = pd.read_csv(filepath_or_buffer=data_path,encoding='utf-8',quoting=1)
                data = dataframe_to_file(path=data_path, format=self.raw_data_format, mode='r')
                logger.info(f"读取 {data_type} 数据共 {data.__len__()} 条from {data_path}")
                data_dict.update({data_type: data})
            elif mode == 'w' and data is not None:
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                # data.to_json(path_or_buf=data_path, orient='records',force_ascii=False,indent=4)
                # data.to_csv(path_or_buf=data_path,encoding='utf-8',quoting=1)
                _ = dataframe_to_file(data=data, path=data_path, mode='w')
                logger.info(f"存储 {data_type} 数据共 {data.__len__()} 条 into {data_path}")
                data_dict.update({data_type: data})

                label_value_counts = data.loc[:, self.label_column].value_counts().to_dict()
                label_value_counts.update({'total': data.__len__()})
                data_statistics.update(({data_type: label_value_counts}))

        data_statistics_path = os.path.join(output_dir, 'data_statistics.json') if output_dir is not None else None
        if mode == 'w' and data_statistics_path:
            dump_obj_as_json_to_file(data_statistics, data_statistics_path)
        return data_dict

    def preprocess_new_data(self) -> pd.DataFrame:
        """
        读取新增的数据，并且划分为 train，dev，test 三个数据集
        """
        raise NotImplementedError

    def combine_data(self, new_add_data_dict, last_version_data_dict):
        if last_version_data_dict == {}:
            new_version_data_dict = new_add_data_dict
        elif new_add_data_dict != {}:
            new_version_data_dict = {}
            for data_type, new_data in new_add_data_dict.items():
                last_type_data = last_version_data_dict.get(data_type)
                new_type_data = pd.concat([last_type_data, new_data], axis=0).reset_index(drop=True)
                new_version_data_dict.update({data_type: new_type_data})

        # 获取支持的类别
        self.support_types_dict = self.get_support_types_dict(new_version_data_dict)

        # 存储数据
        new_version_data_dict = self.read_data(mode='w',
                                               output_dir=self.output_data_dir,
                                               raw_data=new_version_data_dict.get(RAW_DATATYPE),
                                               all_data=new_version_data_dict.get(ALL_DATATYPE),
                                               train_data=new_version_data_dict.get(TRAIN_DATATYPE),
                                               dev_data=new_version_data_dict.get(EVAL_DATATYPE),
                                               test_data=new_version_data_dict.get(TEST_DATATYPE))
        return new_version_data_dict

    @staticmethod
    def create_support_types_dict_from_ls(support_labels_ls) -> Dict[Text,int]:
        support_labels_ls = sorted(support_labels_ls)
        support_types_dict = {label: index for index, label in enumerate(support_labels_ls)}
        return support_types_dict

    @staticmethod
    def create_support_types_dict_from_counter(label_counter:Dict[Text,int],
                                               default_label='其他',
                                               default_label_index= 0,
                                               default_label_count=0,)-> Dict[Text,Dict[Text,int]]:

        support_types_dict = {
            default_label: {SUPPORT_TYPE_INDEX: default_label_index, SUPPORT_TYPE_COUNT: default_label_count}}
        label_index = default_label_index +1
        sorted_label_counter =  sorted(label_counter.items(),key=lambda x:x[1])
        for label, label_count in sorted_label_counter:
            # label_count = support_label_count_series.loc[label]
            support_types_dict.update({label: {SUPPORT_TYPE_INDEX: label_index, SUPPORT_TYPE_COUNT: label_count}})
            label_index += 1
        return  support_types_dict


    def get_support_types_dict(self, new_version_data):
        train_data = new_version_data.get(TRAIN_DATATYPE)
        label_value_counts = train_data.loc[:, self.label_column].value_counts()
        support_labels_ls = []
        for index in label_value_counts.index:
            support_labels_ls.append(str.lower(index))
        support_labels_ls = sorted(support_labels_ls)
        support_types_dict = {label: index for index, label in enumerate(support_labels_ls)}

        dump_json(json_object=support_types_dict, json_path=self.support_types_dict_json_path)
        return support_types_dict

    def train_dev_test_split(self, raw_data: pd.DataFrame, max_threshold=1000,
                             train_size=0.8,
                             dev_size=0.1,
                             test_size=0.1, sample_size_for_label=None,
                             stratify=True) -> Dict[Text, pd.DataFrame]:
        label_values_count = raw_data.loc[:, self.label_column].value_counts()
        # logger.info(f"开始分割数据:{label_values_count}")
        if self.min_threshold:
            # 去除数量较少的类型
            support_label_values_count = label_values_count[label_values_count >= self.min_threshold]
            model_support_labels = set(support_label_values_count.index.tolist())
        # 对部分标签进行过滤
        if self.support_labels:
            model_support_labels = model_support_labels.intersection(self.support_labels)
        # 过滤支持的类型的原始数据
        support_labels_raw_data = raw_data.loc[raw_data.loc[:, self.label_column].isin(model_support_labels)].copy()

        label_values_count = support_labels_raw_data.loc[:, self.label_column].value_counts()
        logger.info(
            f"按照 min_threshold={self.min_threshold}和support_labels={self.support_labels}对意图标注数据进行过滤，共{raw_data.shape}，"
            f"\n按照意图的数据分布为：\n{label_values_count}")
        # 是否对每个类别进行抽样
        sample_size_for_label = sample_size_for_label if sample_size_for_label else self.sample_size_for_label
        if sample_size_for_label:
            sample_data_ls = []
            intents = support_labels_raw_data.intent.value_counts().index.tolist()
            for intent in intents:
                intent_data = support_labels_raw_data.loc[support_labels_raw_data.intent == intent]
                sample_size_for_label = min(intent_data.__len__(), self.sample_size_for_label)
                intent_sample_data = intent_data.sample(sample_size_for_label, random_state=self.random_state)
                sample_data_ls.append(intent_sample_data)

            sample_data = pd.concat(sample_data_ls, axis=0).reset_index(drop=True)
            logger.info(f"抽样数据共 {sample_data.shape},"
                        f"\n按照意图的数据分布为：\n{sample_data.intent.value_counts()}")
        else:
            sample_data = support_labels_raw_data
        # 是否进行分层抽样
        if stratify:
            support_types_dict = self.create_support_types_dict(model_support_labels)
            label_mapping_index = sample_data.loc[:, self.label_column].apply(lambda x: support_types_dict[x])
        else:
            label_mapping_index = None
        train_data, dev_and_test_data = train_test_split(sample_data, train_size=train_size,
                                                         stratify=label_mapping_index,
                                                         random_state=self.random_state)
        if dev_size is not None and dev_size != 0:
            dev_size_in_left = dev_size / (1 - train_size)
            if stratify:
                label_mapping_index = dev_and_test_data.loc[:, self.label_column].apply(lambda x: support_types_dict[x])
            dev_data, test_data = train_test_split(dev_and_test_data, train_size=dev_size_in_left,
                                                   stratify=label_mapping_index,
                                                   random_state=self.random_state)
        else:
            dev_data = None
            test_data = dev_and_test_data

        data_dict = {RAW_DATATYPE: raw_data,
                     ALL_DATATYPE: sample_data, TRAIN_DATATYPE: train_data,
                     EVAL_DATATYPE: dev_data, TEST_DATATYPE: test_data}

        dev_size = 0 if dev_data is None else dev_data.shape[0]

        logger.info("全部数据共{}条,按照 train:dev:test={}:{}:{},分为train={},dev={},test={}"
                    .format(sample_data.shape[0], train_size, dev_size, test_size, train_data.shape[0],
                            dev_size,
                            test_data.shape[0]))
        return data_dict


class DataProcessor():
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, corpus='corpus', dataset='dataset', data_dir=None, output_dir=None,
                 raw_train_data_filename='raw_train.json',
                 raw_dev_data_filename='raw_dev.json',
                 raw_test_data_filename='raw_test.json',
                 batch_strategy= RANDOM_BATCH_STRATEGY,
                 train_batch_size=32, eval_batch_size=32, test_batch_size=32,
                 buffer_size=None, num_parallel_calls=None,
                 use_tpu=False, drop_remainder=False,
                 train_filename="train.json", eval_filename="dev.json", test_filename="test.json",
                 tfrecord_suffix='tf_record',
                 debug=False, debug_example_num=10
                 ):
        self.corpus = corpus
        self.dataset = dataset

        self.debug = debug
        self.debug_example_num = debug_example_num

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.raw_train_data_filename = raw_train_data_filename
        self.raw_dev_data_filename = raw_dev_data_filename
        self.raw_test_data_filename = raw_test_data_filename

        self.buffer_size = tf.data.experimental.AUTOTUNE if buffer_size is None else buffer_size
        self.num_parallel_calls = tf.data.experimental.AUTOTUNE if num_parallel_calls is None else num_parallel_calls
        self.use_tpu = use_tpu
        self.drop_remainder = drop_remainder if drop_remainder is not None or use_tpu else False

        self.batch_strategy = batch_strategy
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size

        self.batch_size_dict = {TRAIN_DATATYPE: self.train_batch_size, EVAL_DATATYPE: self.eval_batch_size,
                                TEST_DATATYPE: self.test_batch_size}

        self.raw_train_filepath = os.path.join(self.data_dir, self.raw_train_data_filename)
        self.raw_eval_filepath = os.path.join(self.data_dir, self.raw_dev_data_filename)
        self.raw_test_filepath = os.path.join(self.data_dir, self.raw_test_data_filename)

        self.train_filepath = os.path.join(self.output_dir, train_filename)
        self.eval_filepath = os.path.join(self.output_dir, eval_filename)
        self.test_filepath = os.path.join(self.output_dir, test_filename)

        self.raw_filepath_dict = {TRAIN_DATATYPE: self.raw_train_filepath, EVAL_DATATYPE: self.raw_eval_filepath,
                                  TEST_DATATYPE: self.raw_test_filepath}

        self.input_filepath_dict = {TRAIN_DATATYPE: self.train_filepath, EVAL_DATATYPE: self.eval_filepath,
                                    TEST_DATATYPE: self.test_filepath}

        self.tfrecord_suffix = tfrecord_suffix

    def prepare_model_data(self, data_type):
        """
        根据 data_type 获取 dataset 或者 generator
        """
        raise NotImplementedError

    def _get_examples(self, data_type='train', output_dataframe=False) -> List[InputExample]:
        """See base class."""
        raise NotImplementedError()

    def _get_train_examples(self, data_dir) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def _get_dev_examples(self, data_dir) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def _get_test_examples(self, data_dir) -> List[InputExample]:
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def process_text(self, text):
        if self.use_spm:
            return tokenization.preprocess_text(text, self.do_lower_case)
        else:
            return tokenization.convert_to_unicode(text)

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            # 读取每一行
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                # 每行为两个 tokens
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
            return lines

    def _decode_record(self, record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        # example = tf.parse_single_example(record, name_to_features)
        example = tf.io.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        if self.use_tpu:
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
        return example

    # def input_fn(self,params):
    #     """The actual input function."""
    #     if use_tpu:
    #         batch_size = params["batch_size"]
    #     else:
    #         batch_size = bsz
    #
    #     # For training, we want a lot of parallel reading and shuffling.
    #     # For eval, we want no shuffling and parallel reading doesn't matter.
    #     d = tf.data.TFRecordDataset(input_file)
    #     if is_training:
    #         d = d.repeat()
    #         d = d.shuffle(buffer_size=100)
    #
    #     d = d.apply(
    #         contrib_data.map_and_batch(
    #             lambda record: _decode_record(record, name_to_features),
    #             batch_size=batch_size,
    #             num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
    #             drop_remainder=drop_remainder))
    #     d = d.prefetch(buffer_size=bsz)
    #
    #     return d

    def get_one_shot_iterator(self, data_type='train', batch_size=1):
        """
        使用 dataset API 来提取数据
        :param batch_size:
        :return:
        """
        input_file = self.input_file_dict[data_type]
        # 载入数据
        dataset = tf.data.TFRecordDataset(input_file)
        dataset = dataset.map(lambda record: self._decode_record(record, self.name_to_features))
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.batch(batch_size)
        # # 创建一个迭代器  遍历数据集并重新得到数据真实值（即只能从头到尾读取一次）
        iterator = dataset.make_one_shot_iterator()
        # 获得包含数据的张量
        next_data = iterator.get_next()
        return next_data

    def get_dataset_from_tfrecord(self, tfrecord_input_file, data_type='train', batch_size=None):
        """
        @time:  2021/6/10 8:55
        @author:wangfc
        @version:
        @description:

        @params:
        @return:
        """
        # input_file = self.input_file_dict[data_type]
        if batch_size is None:
            batch_size = self.batch_size_dict[data_type]
        # 读取 tfrecord数据
        d = tf.data.TFRecordDataset(tfrecord_input_file)

        if data_type == 'train':
            d = d.repeat()  # 生成的序列就会无限重复下去
            d = d.shuffle(buffer_size=self.buffer_size)  #

        # d = d.apply(
        #     tf.data.map_and_batch(
        #         lambda record: self._decode_record(record, self.name_to_features),
        #         batch_size=batch_size,
        #         num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
        #         drop_remainder= self.drop_remainder
        #     ))

        # 解析 tfrecord 数据
        d = d.map(lambda record: self._decode_record(record, self.name_to_features),
                  num_parallel_calls=self.num_parallel_calls)
        # batch
        d = d.batch(batch_size=batch_size, drop_remainder=self.drop_remainder)
        # prefetch
        d = d.prefetch(buffer_size=self.buffer_size)
        return d
