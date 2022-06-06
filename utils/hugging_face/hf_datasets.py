#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/14 16:10 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/14 16:10   wangfc      1.0         None
"""
import os
from typing import Union, Text, List, Optional, Tuple, Dict, Any
from collections import Counter

from tensorflow.python.keras.layers import StringLookup

from data_process.data_generator import KerasDataGenerator, KerasTextClassifierDataGenerator
from utils.io import read_json_file, dump_obj_as_json_to_file
from utils.tensorflow.constants import SEQUENCE
from utils.tensorflow.preprocessing_layers import create_lookup_layer
from utils.time import timeit
import datasets
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
from pprint import pprint
import math

import logging

logger = logging.getLogger(__name__)
# Let's keep it verbose for our tutorial though
from datasets import logging

# logging.set_verbosity_warning()
logging.set_verbosity_info()


class HfDataset():
    def __init__(self, dataset_name, cache_dir=None, num_proc: int = 1,debug=False):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.num_proc = num_proc
        self.debug = debug
        # self.script_path = os.path.join('data_process', 'dataset', 'conll2003.py')

        self.dataset = self.get_dataset()

        self.output_types, self.output_shapes, self.data_signature = None, None, None
        self.lookup_layer = None

    @timeit
    def get_dataset(self) -> Union[datasets.Dataset, datasets.DatasetDict]:
        """

        hugs Face GitHub repo或AWS桶中下载并导入SQuAD python处理脚本(如果它还没有存储在库中)。
        运行SQuAD脚本下载数据集。处理和缓存的SQuAD在一个Arrow 表。
        基于用户要求的分割返回一个数据集。默认情况下，它返回整个数据集。
        """
        try:
            dataset = datasets.load_from_disk(self.cache_dir)
        except Exception as e:
            dataset = datasets.load_dataset(path=self.dataset_name, cache_dir=self.cache_dir)
            dataset.save_to_disk(self.cache_dir)

        # if self.debug:
        #     for key,value in dataset.items():
        #         dataset.update({key:value.select(np.arange(1000))})

        return dataset

    def _get_steps(self, data_type='train', batch_size=64):
        return math.ceil(self.dataset[data_type].__len__() / batch_size)


    @staticmethod
    def _create_padded_batch_dataset(self, dataset, batch_size: int = 64):
        """
        使用 tf.data.Dataset.from_generator tf.data.Dataset
            实现过程 ： generator -> tf.data.Dataset -> .map() ->
            tf2.3.2: tf.data.Dataset.from_generator 不含有 output_signature 参数
            output_types = (tf.int64, tf.int64),
            output_shapes = (tf.TensorShape([]), tf.TensorShape([None])))

            tf2.4.0: 含有 output_signature 参数


        """
        if tf.__version__ < "2.4.0":
            arguments = {"output_types": self.output_types, "output_shapes": self.output_shapes, }
        elif tf.__version__ >= "2.4.0":
            arguments = {"output_signature": self.data_signature}
        # 先创建 generator ,使用 tf.data.Dataset.from_generator 生成 flatmap_dataset
        flatmap_dataset = tf.data.Dataset.from_generator(
            self._create_generator(dataset=dataset), **arguments)
        pprint(list(flatmap_dataset.take(1).as_numpy_iterator()))

        # 使用 map() 对 dataset 进行预处理
        preprocessed_cache_dataset = self._preprocess_dataset(flatmap_dataset)
        pprint(list(preprocessed_cache_dataset.take(1).as_numpy_iterator()))

        # 对 dataset 进行 batch 或者 padded_batch
        padded_batch_dataset = preprocessed_cache_dataset.padded_batch(batch_size=batch_size)

        for batch in padded_batch_dataset.take(1):
            pprint(batch)
        return padded_batch_dataset

    def _create_generator(self, dataset):
        raise NotImplementedError

    def _preprocess_dataset(self, dataset: tf.data.Dataset, batch_size):
        raise NotImplementedError

    def _create_lookup_layer(self):
        lookup_layer = create_lookup_layer(vocabulary=self.vocabulary)
        print(len(lookup_layer.get_vocabulary()))
        return lookup_layer


class Conll2003Dataset(HfDataset):
    """
    https://www.clips.uantwerpen.be/conll2003/ner/

    数据集第一例是单词，第二列是词性，第三列是语法快，第四列是实体标签。
    在NER任务中，只关心第一列和第四列。实体类别标注采用BIO标注法，
    """

    def __init__(self, dataset_name='conll2003', vocab_size=None, *args, **kwargs):
        super(Conll2003Dataset, self).__init__(dataset_name=dataset_name, *args, **kwargs)
        self.script_path = os.path.join('data_process', 'dataset', 'conll2003.py')
        self.vocab_size = vocab_size

        self.dataset = self.get_dataset()

        self.add_pad_tag = True
        self.tags = self._get_tags()

        self.vocabulary,self.max_sequence_length = self._get_vocabulary()
        self.data_signature, self.output_types, self.output_shapes = self._get_data_signature()
        self.lookup_layer = self._create_lookup_layer()

    def explore_dataset(self):
        """
        The returned Dataset object is a memory mapped dataset that behave similarly to a normal map-style dataset.
        It is backed by an Apache Arrow table which allows many interesting features.

        The __getitem__ method will return different format depending on the type of query:
            Items like dataset[0] are returned as dict of elements.
            Slices like dataset[10:20] are returned as dict of lists of elements.
            Columns like dataset['question'] are returned as a list of elements.

        """
        print(self.dataset)
        train_dataset = self.dataset["train"]

        print(f"👉 Dataset len(train_dataset): {len(train_dataset)}")
        print("\n👉 First item 'train_dataset[0]':")
        pprint(train_dataset[0])

        # Or get slices with several examples:
        print("\n👉Slice of the two items 'dataset[10:12]':")
        pprint(train_dataset[10:12])

        # You can get a full column of the dataset by indexing with its name as a string:
        print(train_dataset['tokens'][:10])

        # You can inspect the dataset column names and types
        print("Column names:")
        pprint(train_dataset.column_names)
        print("Features:")
        pprint(train_dataset.features)

        # Get a sample of train data and print it out:
        for item in self.dataset["train"]:
            sample_tokens = item['tokens']
            sample_tag_ids = item["ner_tags"]
            print(sample_tokens)
            print(sample_tag_ids)
            break
        return sample_tokens, sample_tag_ids

    def _get_tags(self) -> List[Text]:
        if self.dataset is None:
            self.dataset = self.get_dataset()

        sample_tokens, sample_tag_ids = self.explore_dataset()
        raw_tags_filename = os.path.join('corpus', 'conll2003', 'raw_tags.json')
        if os.path.exists(raw_tags_filename):
            raw_tags = read_json_file(json_path=raw_tags_filename)
        else:
            # The dataset also give the information about the mapping of NER tags and ids:
            dataset_builder = datasets.load_dataset_builder('conll2003', cache_dir=self.cache_dir)
            raw_tags = dataset_builder.info.features['ner_tags'].feature.names
            print(raw_tags)
            dump_obj_as_json_to_file(filename=raw_tags_filename, obj=raw_tags)

        sample_tags = [raw_tags[i] for i in sample_tag_ids]
        print(list(zip(sample_tokens, sample_tags)))
        # Add a special tag <PAD> to the tag set, which is used to represent padding in the sequence.
        if self.add_pad_tag:
            tags = ['<PAD>'] + raw_tags
        else:
            tags = raw_tags
        print(tags)
        return tags

    def _get_data_signature(self):

        output_types = (tf.string, tf.int32)
        # tf.TensorShape([]): 标量, tf.TensorShape([None])): array
        output_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))

        data_signature = (
            tf.TensorSpec(shape=(None,), dtype=output_types[0]),
            tf.TensorSpec(shape=(None,), dtype=output_types[1])
        )

        return data_signature, output_types, output_shapes

    def _get_vocabulary(self) -> List[Text]:
        """
        参考  MovielensRecommendationTask._get_vocabulary()
        from tasks.recommendation_task import MovielensRecommendationTask
        1.  Modifying the dataset example by example
        dataset.map()
            The function you provide to .map() should accept an input with the format of an item of the dataset:
            function(dataset[0]) and return a python dict.
            The columns and type of the outputs can be different than the input dict.
            In this case the new keys will be added as additional columns in the dataset.
            Bascially each dataset example dict is updated with the dictionary returned by the function like this:
            example.update(function(example))

        """
        train_dataset = self.dataset["train"]
        # 使用 map 增加一列 lower_tokens
        lower_train_dataset = train_dataset.map(lambda example:
                                                {"lower_tokens": [token.lower() for token in example["tokens"]]},
                                                num_proc=self.num_proc)

        # 获取训练的 train_tokens
        train_tokens = lower_train_dataset['lower_tokens']
        counter = Counter()
        max_sequence_length = 0
        for tokens in train_tokens:
            counter.update(tokens)
            if tokens.__len__() > max_sequence_length:
                max_sequence_length = tokens.__len__()

        most_common_vocabulary = counter.most_common(n=self.vocab_size)
        vocabulary = []
        for token, c in most_common_vocabulary.__iter__():
            vocabulary.append(token)
        # train_tokens = tf.ragged.constant(self.dataset["train"]["tokens"])
        # train_tokens = tf.map_fn(tf.strings.lower, train_tokens)
        # train_tokens = train_tokens.batch(1_000).cache()
        # vocabulary = np.unique(np.concatenate(list(train_tokens)))
        return vocabulary,max_sequence_length

    def _create_generator(self, dataset: datasets.Dataset):
        def data_generator():
            for item in dataset:
                yield item['tokens'], item['ner_tags']

        return data_generator


    def _preprocess(self,tokens, tag_ids):
        def preprecess_tokens(tokens):
            tokens = tf.strings.lower(tokens)
            return tokens

        preprocessed_tokens = preprecess_tokens(tokens)
        # increase by 1 for all tag_ids,
        # because `<PAD>` is added as the first element in tags list
        # preprocessed_tag_ids = tag_ids + 1
        preprocessed_tag_ids = tf.math.add(tag_ids, 1)
        return preprocessed_tokens, preprocessed_tag_ids

    def _preprocess_dataset(self, dataset: tf.data.Dataset):
        # With `padded_batch()`, each batch may have different length
        # shape: (batch_size, None)
        dataset = dataset.map(self._preprocess).cache()
        return dataset



    def _preprocee_features(self,example:Dict[Text,Any], features:List[Text] = ['tokens', 'ner_tags'] ):
        """
        抽取特征进行预处理，后面使用 .map()
        """
        # 获取原始特征
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        # 进行 preprocessing
        preprocessed_tokens, preprocessed_tag_ids = self._preprocess(tokens=tokens,tag_ids=ner_tags)
        token_ids = self.lookup_layer(preprocessed_tokens)
        example['token_ids'] = token_ids
        example['tag_ids']= preprocessed_tag_ids
        return example


    def extract_features(self,dataset:datasets.Dataset):
        """
        1. 对 Dataset or batch Dataset 抽取特征并对其进行预处理
        2. 对全部是 0 的数据进行过滤，否则lstm 和 crf层会报错
        https://github.com/tensorflow/tensorflow/issues/33069
        tensorflow lstm CUDNN_STATUS_BAD_PARAM   cudnnSetRNNDataDescriptor()
        CRF layer do not support left padding

        """
        dataset = dataset.map(self._preprocee_features)

        # 对全部是 0 的数据进行过滤，否则lstm 和 crf层会报错
        dataset = dataset.filter(lambda example:np.sum(example['token_ids']) >0)
        # tokens_ls, tag_ids_ls= [],[]
        # for item in dataset:
        return dataset










class NerBatchDataGenerator(KerasTextClassifierDataGenerator):
    def __init__(self, dataset: datasets.Dataset,
                 epochs: int = 1,*args,**kwargs
  ):
        super(NerBatchDataGenerator, self).__init__(data=dataset,*args,**kwargs,run_epoch_end=False)
        self._epochs = epochs
        # we use `on_epoch_end` method to prepare data for the next epoch
        # set current epoch to `-1`, so that `on_epoch_end` will increase it to `0`
        self._current_epoch = -1
        self.on_epoch_end()


    def __getitem__(self, index):
        """
        重装 __getitem__ 函数，获取每个batch
        """
        # on_epoch_end(), 根据不同的 strategy,Generate indexes of the batch
        # batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        start = index * self._current_batch_size
        end = start + self._current_batch_size

        # return input and target data, as our target data is inside the input
        # data return None for the target data
        return self.prepare_batch(self._data, start, end)

    def prepare_batch(self,data:datasets.Dataset,start:int,end:int):
        # 使用 Dataset 的index 后期 batch_data
        batch_data = data[start:end]
        batch_token_id = batch_data['token_ids']
        batch_tag_ids = batch_data['tag_ids']

        # 如果不对 batch 做 padding，batch 中每条数据的长度不一致
        # 对 token 和 tag_id 进行 padding
        padded_batch_token_ids = pad_sequences(batch_token_id,padding='post')
        padded_batch_tag_ids = pad_sequences(batch_tag_ids,padding='post')

        return (padded_batch_token_ids,padded_batch_tag_ids)


    def on_epoch_end(self) -> None:
        """Update the data after every epoch."""
        self._current_epoch += 1
        # _current_batch_size 保持不变
        self._current_batch_size = self.batch_size  # self._linearly_increasing_batch_size()
        # _data 等于 self.data
        self._data = self.data  # self._shuffle_and_balance(self._current_batch_size)

        self.indexes = np.arange(len(self._data))
        if self.shuffle:
            np.random.shuffle(self.indexes)



def create_ner_data_generators(
    dataset: Conll2003Dataset,
    batch_sizes: Union[int, List[int]],
    epochs: int,
    batch_strategy: Text = SEQUENCE,
    eval_num_examples: int = 0,
    random_seed: Optional[int] = None,
    shuffle: bool = True,
    debug=False,
) -> Tuple[NerBatchDataGenerator, Optional[NerBatchDataGenerator]]:
    """Create data generators for train and optional validation data.

    Args:
        model_data: The model data to use.
        batch_sizes: The batch size(s).
        epochs: The number of epochs to train.
        batch_strategy: The batch strategy to use.
        eval_num_examples: Number of examples to use for validation data.
        random_seed: The random seed.
        shuffle: Whether to shuffle data inside the data generator.

    Returns:
        The training data generator and optional validation data generator.
    """
    validation_data_generator = None
    if eval_num_examples > 0:
        evaluation_model_data = dataset.dataset['eval']
        # 抽取特征并进行预处理
        evaluation_model_data = dataset.extract_features(evaluation_model_data)
        validation_data_generator = NerBatchDataGenerator(
            evaluation_model_data,
            batch_size=batch_sizes,
            epochs=epochs,
            batch_strategy=batch_strategy,
            shuffle=shuffle,
        )

    train_data = dataset.dataset['train']
    # if debug:
    #     #  使用 index 生产 dict
    #     # train_data = train_data[:1000]
    #     train_data = train_data.select(np.arange(1000))
    # 抽取特征并进行预处理
    train_data = dataset.extract_features(train_data)

    data_generator = NerBatchDataGenerator(
        train_data,
        batch_size=batch_sizes,
        epochs=epochs,
        batch_strategy=batch_strategy,
        shuffle=shuffle,
    )

    return data_generator, validation_data_generator
