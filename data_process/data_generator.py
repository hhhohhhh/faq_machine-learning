#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/6/16 17:23 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/16 17:23   wangfc      1.0         None
"""
from typing import List, Any, Tuple

import math
import tensorflow as tf
from bert4keras.snippets import sequence_padding, is_string
import numpy as np
from data_process.data_analysis import TextClassifierDataAnalysis
from data_process.data_example import InputExample
from data_process.data_feature import convert_single_example
from data_process.dataset.batch_strategies import Balanced_Batch_Strategy
from .dataset.hsnlp_faq_dataset import FAQSentenceExample
from utils.constants import BALANCED_BATCH_STRATEGY, RANDOM_BATCH_STRATEGY, SEQUENCE_BATCH_STRATEGY, TRAIN_DATATYPE
import logging

logger = logging.getLogger(__name__)


def demo_generator(batch_size=4):
    """
    # loading data
    X_train, Y_train = load_data(...)

    # data processing
    # ................
    x_train=None
    y_train =None

    total_size = X_train.size
    # batch_size means how many data you want to train one step
    """
    total_size = 101
    x_train = np.arange(total_size).tolist()
    y_train = np.random.randint(low=0, high=3, size=total_size).tolist()
    while True:
        for i in range(total_size // batch_size):
            yield x_train[i * batch_size:(i + 1) * batch_size], y_train[i * batch_size:(i + 1) * batch_size]


class DataGenerator(object):
    """数据生成器模版

    增加 batch_strategy:
    """

    def __init__(self, data, tokenizer, max_seq_length, data_type='train',
                 batch_size=32, batch_strategy=None,
                 buffer_size=None):
        self.data_type = data_type
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __repr__(self):
        return f"{self.__class__.__name__} {self.data_type} with length={self.data.__len__()}, batches={self.steps}," \
               f"batch_size={self.batch_size}"

    def __len__(self):
        return self.steps

    def forfit(self, datatype='train', init_epoch=0):
        """
        训练的时候使用
        while True : 则该 generator 会永不停止
        """
        epoch = init_epoch
        if datatype == 'train':
            random = True
        else:
            random = False

        logger.info(f"\n开始生成datatype={datatype} epoch={epoch} random={random} 的data generator")

        while True:
            # 每次调用__iter__ 函数，迭代生成一个batch
            for batch in self.__iter__(random, epoch=epoch):
                yield batch

            # 直到每个 epoch 迭代完成
            logger.info(f"完成datatype={datatype} epoch={epoch} 的data generator")
            epoch += 1

    def for_predict(self):
        """
        预测的时候使用，生成批数据
        """
        # while True:
        for d in self.__iter__(random=False, with_example=True):
            yield d

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:
                # 随机生成 data中的一个数据的 生成器
                def generator():
                    for i in np.random.permutation(len(self.data)):
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        # d_current = next(data)
        for d_current in data:
            yield False, d_current
            # d_current = d_next
        yield True, None

    def __iter__(self, random=False, epoch=None, with_example=False):
        raise NotImplementedError

    def to_dataset(self, types, shapes, names=None, padded_batch=False):
        """转为tf.data.Dataset格式
        如果传入names的话，自动把数据包装成dict形式。
        """
        if names is None:
            generator = self.forfit
        else:

            if is_string(names):
                warps = lambda k, v: {k: v}
            elif is_string(names[0]):
                warps = lambda k, v: dict(zip(k, v))
            else:
                warps = lambda k, v: tuple(
                    dict(zip(i, j)) for i, j in zip(k, v)
                )

            def generator():
                for d in self.forfit():
                    yield warps(names, d)

            types = warps(names, types)
            shapes = warps(names, shapes)

        if padded_batch:
            dataset = tf.data.Dataset.from_generator(
                generator, output_types=types
            )
            dataset = dataset.padded_batch(self.batch_size, shapes)
        else:
            dataset = tf.data.Dataset.from_generator(
                generator, output_types=types, output_shapes=shapes
            )
            dataset = dataset.batch(self.batch_size)

        return dataset


class TextClassifierDataGenerator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False, padding=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.max_seq_length)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                if padding:
                    # 在生成 input的时候，需要做 padding 到最大的长度，否则精度训练不大好
                    batch_token_ids = sequence_padding(batch_token_ids, length=self.max_seq_length)
                    batch_segment_ids = sequence_padding(batch_segment_ids, length=self.max_seq_length)
                    batch_labels = sequence_padding(batch_labels)
                else:
                    pass

                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class EventClassifierDataGenerator(TextClassifierDataGenerator):
    """数据生成器,生成 批数据
    """

    def __iter__(self, random=False, with_example=False):
        batch_token_ids_ls, batch_segment_ids_ls, batch_labels_ls = [], [], []
        batch_examples = []
        # data_iterator = self.sample(random)
        for is_end, example in self.sample(random):
            if not is_end:
                # example 是一个 JuyuanNerClassifierPipelineInputExampleV2 实例
                guid = example.guid
                entity_index = example.entity_index
                title = example.title
                content = example.content
                event_label_ls = example.event_label_ls
                # print(f"guid={guid},entity_index={entity_index}")

                token_ids = example.input_ids
                segment_ids = example.segment_ids
                label_ids = example.label_ids
                # token_ids, segment_ids = self.tokenizer.encode(context, maxlen=self.max_seq_length)

                batch_token_ids_ls.append(token_ids)
                batch_segment_ids_ls.append(segment_ids)
                batch_labels_ls.append([label_ids])
                batch_examples.append(example)

            if len(batch_token_ids_ls) == self.batch_size or (is_end and len(batch_token_ids_ls) > 0):
                # 在生成 input的时候，需要做 padding 到最大的长度，否则精度训练不大好
                batch_token_ids = sequence_padding(batch_token_ids_ls, length=self.max_seq_length)
                batch_segment_ids = sequence_padding(batch_segment_ids_ls, length=self.max_seq_length)
                batch_labels = np.array(batch_labels_ls).squeeze(axis=1)
                if not with_example:
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                else:
                    yield [batch_token_ids, batch_segment_ids], batch_labels, batch_examples
                batch_token_ids_ls, batch_segment_ids_ls, batch_labels_ls = [], [], []


class IntentClassifierDataGenerator(TextClassifierDataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False, epoch=0):
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_label_id = [], [], [], []
        batch_example_guids = []

        ex_index = 0
        batch_index = 0
        if epoch == 0:
            if_print = True
        else:
            if_print = False

        #
        for is_end, example in self.sample(random):
            if not is_end:
                feature = convert_single_example(ex_index=ex_index, example=example,
                                                 max_seq_length=self.max_seq_length,
                                                 tokenizer=self.tokenizer,
                                                 if_print=if_print)
                # hf_transformer的 输入
                batch_input_ids.append(feature.input_ids)
                batch_attention_mask.append(feature.input_mask)
                batch_token_type_ids.append(feature.segment_ids)
                batch_label_id.append([feature.label_id])
                batch_example_guids.append(example.guid)
                ex_index += 1
            if len(batch_input_ids) == self.batch_size or (is_end and batch_input_ids.__len__() > 0):
                # 转换为 np array
                input_ids_na = np.array(batch_input_ids).astype(np.int64)
                attention_mask_na = np.array(batch_attention_mask).astype(np.int64)
                token_type_ids_na = np.array(batch_token_type_ids).astype(np.int64)
                label_id_na = np.array(batch_label_id).astype(np.int64)
                logger.debug(f"epoch={epoch},batch_index={batch_index},batch_example_guids={batch_example_guids}")
                yield [input_ids_na, attention_mask_na, token_type_ids_na], label_id_na
                batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_label_id = [], [], [], []
                batch_example_guids = []
                batch_index += 1

    def sample(self, random=False) -> FAQSentenceExample:
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:
                def generator():
                    for i in np.random.permutation(len(self.data)):
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        # d_current = next(data)
        for d_current in data:
            yield False, d_current
            # d_current = d_next
        # data 结束的时候，输出 is_end= True
        yield True, None


class KerasDataGenerator(tf.keras.utils.Sequence):

    def __len__(self) -> int:
        """Number of batches in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Gets batch at position `index`.

        Arguments:
            index: position of the batch in the Sequence.

        Returns:
            A batch (tuple of input data and target data).
        """
        raise NotImplementedError

    def on_epoch_end(self) -> None:
        """Update the data after every epoch."""
        raise NotImplementedError



class KerasTextClassifierDataGenerator(KerasDataGenerator):
    """
    from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    继承 tf.keras.utils.Sequence

    主要重写 : convert_single_example_to_feature()
            __data_generation(self, indexes)： 根据 index 生成数据的函数，继承类需要重写该函数
    """

    def __init__(self, data: List[InputExample]=None,
                 tokenizer=None,
                 max_seq_length: int = 128,
                 data_type='train',id_key='guid', feature_key='text_a', label_key='label_id',
                 batch_size=32, batch_strategy=RANDOM_BATCH_STRATEGY,
                 shuffle=True, seed=12345,
                 examples:List[Any]=None,
                 data_analysis:TextClassifierDataAnalysis =None,
                 run_epoch_end=True
                 ):
        """
        @time:  2021/9/10 9:26
        @author:wangfc
        @version:
        @description: 'Initialization'

        on_epoch_end(): 根据不同的 strategy,获取每个 epoch 数据的索引 indexes
        __getitem__: Generate one batch of data
        __batch_data_generation(): 生成具体的每个 batch
        @return:
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data_type = data_type

        self.id_key = id_key
        self.feature_key = feature_key
        self.label_key = label_key

        self.batch_strategy = batch_strategy
        self.seed = seed

        self.data_analysis = data_analysis

        if self.data_type == TRAIN_DATATYPE and self.batch_strategy == BALANCED_BATCH_STRATEGY:
            # 用于 batch_strategy
            # self.data_analysis = TextClassifierDataAnalysis(data_type=data_type, data_examples=data)

            self.balanced_batch_strategy = Balanced_Batch_Strategy(data_analyser=self.data_analysis,
                                                                   batch_size=batch_size,
                                                                   shuffle=shuffle,
                                                                   seed=seed)
        self.batch_size = batch_size
        self.shuffle = shuffle
        # 主要使用回复测试数据
        self.examples = examples

        self.epoch_index = 0

        self.ex_index = 0  # 输入的 example 个数
        self.batch_index = 0
        if run_epoch_end:
            self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, batch_index):
        """
        Generate one batch of data：
        __getitem__ method should return (X, y) value pair where X represents the input and y represents the output.
        The input and output can take any shape with [batch_size, ...] inside
        """
        # on_epoch_end(), 根据不同的 strategy,Generate indexes of the batch
        batch_indexes = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        # 根据不同的 indexes , Generate data
        batch_x, batch_y = self.__batch_data_generation(batch_indexes)

        return batch_x, batch_y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        使用不同的 batch_strategy， 获取每个 epoch 数据的索引 indexes
        """
        self.indexes = np.arange(len(self.data))
        if self.data_type == TRAIN_DATATYPE:
            # 对于训练集可以选择不同 batch_strategy
            if self.batch_strategy == RANDOM_BATCH_STRATEGY and self.shuffle == True:
                # 对 self.indexes 进行重排（inplace）
                # Shuffling the order in which examples are fed to the classifier is helpful so that batches between epochs do not look alike.
                # Doing so will eventually make our model more robust.
                np.random.shuffle(self.indexes)
            elif self.batch_strategy == SEQUENCE_BATCH_STRATEGY:
                # 按照顺序进行输出
                pass
            elif self.batch_strategy == BALANCED_BATCH_STRATEGY:
                # 根据 BALANCED_BATCH_STRATEGY 获取 indexes
                self.indexes = self.balanced_batch_strategy.balanced_sampling(epoch=self.epoch_index)


        # 自动增加 self.epoch_no
        self.epoch_index += 1

    # def convert_single_example_to_feature(self, example):
    #     """
    #     原始的 根据 index 生成数据的函数，继承类需要重写该函数
    #     """
    #     return example.__getattribute__(self.feature_key)

    # def __batch_data_generation(self, batch_indexes):
    #     """Generates data containing batch_size samples
    #
    #     """  # X : (n_samples, *dim, n_channels)
    #     # Generate batch data
    #     batch_x = []
    #     batch_y = []
    #     for i, index in enumerate(batch_indexes):
    #         example = self.data[index]
    #         x = self.convert_single_example_to_feature(example)
    #         y = example.__getattribute__(self.label_key)
    #         # 每次生成一个 example，增加一个数量
    #         self.ex_index += 1
    #         batch_x.append(x)
    #         batch_y.append(y)
    #     # 自动增加 batch_index
    #     self.batch_index += 1
    #     return batch_x, batch_ya_generation(batch_indexes)
    # return batch_x, batch_y

    def _convert_single_example_to_feature(self, example:InputExample):
        feature = convert_single_example(ex_index=self.ex_index, example=example,
                                         max_seq_length=self.max_seq_length,
                                         tokenizer=self.tokenizer)
        return feature

    def __batch_data_generation(self, batch_indexes):
        """Generates data containing batch_size samples

        """  # X : (n_samples, *dim, n_channels)
        INPUT_FEATURE_KEYS = ['input_ids', 'input_mask', 'segment_ids']

        # Generate batch data
        batch_examples = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_label_id = []
        for i, index in enumerate(batch_indexes):
            example = self.data[index]
            feature = self._convert_single_example_to_feature(example)
            batch_examples.append(example)
            input_ids = feature.__getattribute__('input_ids')
            input_mask = feature.__getattribute__('input_mask')
            segment_ids = feature.__getattribute__('segment_ids')

            y = example.__getattribute__(self.label_key)
            # 每次生成一个 example，增加一个数量
            self.ex_index += 1

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(input_mask)
            batch_token_type_ids.append(segment_ids)
            batch_label_id.append(y)

        input_ids_na = np.array(batch_input_ids).astype(np.int64)
        attention_mask_na = np.array(batch_attention_mask).astype(np.int64)
        token_type_ids_na = np.array(batch_token_type_ids).astype(np.int64)
        label_id_na = np.array(batch_label_id).astype(np.int64)

        batch_x, batch_y = [input_ids_na, attention_mask_na, token_type_ids_na], label_id_na
        batch_ids = [example.__getattribute__('guid') for example in batch_examples]

        # 多标签的时候会报错
        # batch_labels_count = Counter(batch_label_id)
        # logger.debug(
        #     f"epoch_index={self.epoch_index},batch_index={self.batch_index},batch_labels_count={batch_labels_count},batch_ids={batch_ids}")

        # 自动增加 batch_index
        self.batch_index += 1

        return batch_x, batch_y


if __name__ == '__main__':
    for x, y in demo_generator():
        print(f"x={x},y ={y}")
