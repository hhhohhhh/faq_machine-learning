#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/10 9:06 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/10 9:06   wangfc      1.0         None
"""
import random
from collections import Counter
from typing import List, Dict, Text, Set
import math
import numpy as np

from data_process.data_analysis import TextClassifierDataAnalysis
from data_process.data_example import InputExample

import logging

logger = logging.getLogger(__name__)


class Balanced_Batch_Strategy():
    def __init__(self,
                 data_analyser: TextClassifierDataAnalysis,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 seed=1234
                 ):
        # 统计信息
        self.data_analyser = data_analyser

        #  获取 label 与其 数据的 index 对应的字典
        self.label_to_indexes_mapping = self.data_analyser.label_to_indexes_mapping.copy()
        self.index_to_label_mapping = self.data_analyser.index_to_label_mapping.copy()
        self.label_to_proportion_mapping = self.data_analyser.label_to_proportion_mapping.copy()

        # 对于 label 比较多的场景， batch_size 需要设置足够大，才能保证每个 batch 中含有数量较少的类型
        self.batch_size = batch_size
        # 计算每个batch中 label 对应的个数
        sorted_label_to_sample_size_per_batch = sorted([(label, self.batch_size * proportion) for label, proportion in
                                                        self.label_to_proportion_mapping.items()],
                                                       key=lambda x: x[1], reverse=True)
        self.label_to_sample_size_per_batch = {label: sample_size_per_batch for label, sample_size_per_batch in
                                               sorted_label_to_sample_size_per_batch}
        self.shuffle = shuffle
        self.seed = seed
        logger.info(f"batch_size={batch_size},label_to_sample_size_per_batch:{self.label_to_sample_size_per_batch}")

    def balanced_sampling(self, epoch: int):
        """
        This algorithm distributes classes across batches to balance the data set.
        To prevent oversampling rare classes and undersampling frequent ones,
        it keeps the number of examples per batch roughly proportional to the relative number of examples
        in the overall data set.
        """

        label_to_indexes_mapping = self.label_to_indexes_mapping.copy()

        sorted_label_to_proportion_mapping = sorted(self.label_to_proportion_mapping.items(), key=lambda x: x[1])

        balanced_indexes = []
        # 进行抽样
        if self.seed:
            seed = epoch + self.seed
            random.seed(seed)
            np.random.seed(seed=seed)

        # 抽样每个 batch的时候，根据 label 对应的比例选取 对应的 index
        sampled_indexes_num = 0
        batch_index = 0
        balanced_batch_sampled_labels_size = []
        try:
            while any([indexes.__len__() > 0 for label, indexes in label_to_indexes_mapping.items()]):
                sampled_label_indexes_per_batch = []
                # 循环对 label 进行抽样
                for label, label_sample_size_per_batch_size in self.label_to_sample_size_per_batch.items():
                    # 获取 label 对应的 index
                    label_indexes_population = label_to_indexes_mapping[label]

                    # 根据 label_sample_size_per_batch_size 可以是带小数部分的数值转换为整数
                    label_sample_size_int = math.floor(label_sample_size_per_batch_size)
                    label_sample_size_decimal = label_sample_size_per_batch_size - label_sample_size_int
                    # 对小数部分进行按照比例抽样
                    label_sample_size_decimal_round = 1 if random.random() < label_sample_size_decimal else 0
                    label_sample_size_per_batch_size_int = label_sample_size_int + label_sample_size_decimal_round

                    if label_sample_size_per_batch_size_int > label_indexes_population.__len__():
                        # 当 sample_size  超过剩下的 label_index_population
                        label_sample_size_per_batch_size_int = label_indexes_population.__len__()

                    if label_sample_size_per_batch_size_int == 0 or label_indexes_population.__len__() == 0:
                        # 当 label_index_population 为空的时候
                        continue

                    sampled_label_indexes = random.sample(population=label_indexes_population,
                                                          k=label_sample_size_per_batch_size_int)
                    # 计算剩下的 label_index_population
                    left_label_index_population = label_indexes_population.difference(set(sampled_label_indexes))
                    # 更新 label_to_index_mapping
                    label_to_indexes_mapping.update({label: left_label_index_population})
                    sampled_label_indexes_per_batch.extend(sampled_label_indexes)

                # 更新已经被抽样的index的数量
                sampled_indexes_num += sampled_label_indexes_per_batch.__len__()
                # 可能存在多余或者少于 batch_size 的情况
                if sampled_label_indexes_per_batch.__len__() > self.batch_size:
                    indexes_per_batch = np.random.choice(sampled_label_indexes_per_batch, size=self.batch_size,
                                                         replace=False)
                elif sampled_label_indexes_per_batch.__len__() < self.batch_size:
                    add_size = self.batch_size - sampled_label_indexes_per_batch.__len__()
                    left_index_size = np.sum([indexes.__len__() for label, indexes in label_to_indexes_mapping.items()])
                    if left_index_size < add_size + 1:
                        # 判断是否剩余的 index 都可以加入
                        add_indexes = set()
                        for label, indexes in label_to_indexes_mapping.items():
                            add_indexes = add_indexes.union(indexes)
                            # 更新 label_to_indexes_mapping
                            label_to_indexes_mapping.update({label: set()})
                        # 更新已经被抽样的index的数量
                        sampled_indexes_num += left_index_size
                    else:
                        add_indexes = np.random.choice(sampled_label_indexes_per_batch, size=add_size, replace=True)
                    indexes_per_batch = sampled_label_indexes_per_batch + list(add_indexes)
                else:
                    indexes_per_batch = sampled_label_indexes_per_batch

                if self.shuffle:
                    # 是否对每个 batch 进行 shuffle
                    np.random.shuffle(indexes_per_batch)

                batch_sampled_labels_size = self._data_indexes_statistics(batch_index=batch_index,
                                                                          sampled_label_indexes_per_batch=sampled_label_indexes_per_batch,
                                                                          data_indexes=indexes_per_batch,
                                                                          sampled_indexes_num = sampled_indexes_num,
                                                                          sorted_label_to_proportion_mapping=sorted_label_to_proportion_mapping)

                balanced_indexes.extend(indexes_per_batch)
                balanced_batch_sampled_labels_size.append(batch_sampled_labels_size)
                batch_index += 1
        except Exception as e:
            logger.error(f"平衡抽样的时候发生错误： {e}", exc_info=True)

        assert sampled_indexes_num == self.index_to_label_mapping.__len__()
        95637 + 229 == 95866

        # 每次循环后的 标签比例
        self._data_indexes_statistics(data_indexes=balanced_indexes,sampled_indexes_num =sampled_indexes_num,
                                      sorted_label_to_proportion_mapping=sorted_label_to_proportion_mapping)

        balanced_batch_sampled_labels_size_count = Counter(balanced_batch_sampled_labels_size)
        logger.info(
            f"使用 balanced_strategy 抽样之后的 labels_size 统计：{balanced_batch_sampled_labels_size_count},\n每个batch的 labels_size ={balanced_batch_sampled_labels_size}")
        return balanced_indexes

    def _data_indexes_statistics(self, batch_index=None, sampled_label_indexes_per_batch=None,sampled_indexes_num=None,
                                 data_indexes: List[int] = None,
                                 sorted_label_to_proportion_mapping=None) -> int:
        # debug 信息
        sampled_labels = [self.index_to_label_mapping[index] for index in data_indexes]
        # 每次循环后的 标签对应的数据数量
        sorted_sampled_labels_counter = {label: count for label, count in
                                         sorted(Counter(sampled_labels).items(),
                                                key=lambda x: x[1])}
        # 每次循环后的 标签个数
        sampled_labels_size = sorted_sampled_labels_counter.keys().__len__()
        sorted_sampled_labels_proportion = {label: count / data_indexes.__len__()
                                            for label, count in sorted_sampled_labels_counter.items()}
        if sampled_label_indexes_per_batch:
            sampled_label_indexes_per_batch_size = sampled_label_indexes_per_batch.__len__()
            logger.debug(f"batch_index={batch_index},该batch 中抽样的index 数量={sampled_label_indexes_per_batch_size},"
                         f"总共已经抽样的 sampled_indexes_num = {sampled_indexes_num}")

        logger.debug(f"抽样后的label数量={sampled_labels_size},\n"
                     f"sorted_sampled_labels_counter={sorted_sampled_labels_counter},\n"
                     f"sorted_sampled_labels_proportion={sorted_sampled_labels_proportion},\n"
                     f"sorted_label_to_proportion_mapping={sorted_label_to_proportion_mapping}")
        return sampled_labels_size


class Balanced_Batch_StrategyV2(Balanced_Batch_Strategy):
    """
    @time:  2022/1/6 11:48
    @author:wangfc
    @version: 新增 multilabel_classifier 多标签的情况
    @description:

    @params:
    @return:
    """

    def __init__(self, multilabel_classifier=True, *args, **kwargs):
        super(Balanced_Batch_StrategyV2, self).__init__(*args, **kwargs)
        self.multilabel_classifier = multilabel_classifier
