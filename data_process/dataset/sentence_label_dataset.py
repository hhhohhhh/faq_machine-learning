#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/3/5 15:18 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/5 15:18   wangfc      1.0         None

Dataset: 分为 Dataset 和 IterableDataset 两种
Dataset：为 map-style 的Dataset， 使用 sampler 来抽样选择 index
IterableDataset： 为 iterable-styel 的Dataset，其抽样的方式完全由 __iter__() 函数决定
因此，对于 hard online triplet learning 的时候 ，同一批数据 batch data，需要有 K个相同标签的数据，可以选择使用  IterableDataset

"""
import bisect
from typing import List
import numpy as np

from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from data_process import BasicDataset

import logging
logger = logging.getLogger(__name__)


class SentenceLabelExample(object):
    """
    @author:wangfc
    @desc: 定义每个 TripletExample 类
    @version：
    @time:2021/3/2 17:30

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self, guid:int=None,id:int=None,text:str=None,label:int=None):
        self.guid=guid
        self.id = id
        self.text = text
        self.label = label


class SentenceLabelFeature():
    def __init__(self, guid:int=None,id:int=None,text:str=None,label:int=None):
        self.guid=guid
        self.id = id
        self.text = text
        self.label = label

    def to_dict(self):
        output_dict = {}
        # return dict(guid=self.guid,id=self.id, text=self.text,label=self.label)
        # 过滤 非 None的key
        for key,value in self.__dict__.items():
            if value is not None:
                output_dict.update({key:value})
        return output_dict



class SentenceLabelDatasetV1(BasicDataset):
    """
    主要代码源自：sentence-transformers v0.3.8
    自定义 SentenceLabelDataset，生成 (sentence,label) pair,
    而每个 batch中 数据应该满足： 具有相同标签的数据可以在同一个 batch之中，从而为后面做 online triplet mining 做准备
    否则，如果同一个batch 中的 label都不同的话，每条数据找不到 positive

    """
    def __init__(self,features:SentenceLabelFeature,data_type='train',
                 provide_positive=True,provide_negative=True,K=4):
        self.features = features
        self.data_type = data_type

        self.provide_positive =provide_positive
        self.provide_negative =provide_negative
        self.K=4


        # 对所有 features 按照 label 分组，并记录分组的信息
        # 记录分组后的 features
        self.grouped_features = []
        # 记录分组后的 右边届
        self.groups_right_border = []
        # 记录分组后的 example index
        self.grouped_indexes = []


    def __getitem__(self, item)->SentenceLabelFeature:
        """
        @author: wangfc27441
        @desc:   在做 online_hard_triplet 的时候需要每个 batch 都要有 PK 个样本：P 张不同的 脸/句子 ，每个对应都对应的 K个相似的 脸/句子
            # 这儿选取 K =4

        @version：
        @time:2020/10/26 17:59

        """

        if not self.provide_positive and not self.provide_negative:
            return [self.grouped_features[item]], self.grouped_labels[item]
        # 获取 anchor
        anchor, anchor_label = self.get_feature(item)
        # 获取边界
        left_border, right_border = self.get_border(item)
        if self.provide_positive:
            positive_item_idx = np.random.choice(
                np.concatenate([self.idxs[left_border:item], self.idxs[item + 1:right_border]]))
            positive_text = self.groups_texts[positive_item_idx]
            # positive = self.grouped_inputs[positive_item_idx]
            positive_label = self.grouped_labels[positive_item_idx]
            assert positive_label == anchor_label
        else:
            positive = []

        if self.provide_negative and not self.get_hard_triplet:
            negative_item_idx = np.random.choice(
                np.concatenate([self.idxs[0:left_border], self.idxs[right_border:]]))
            negative_text = self.groups_texts[negative_item_idx]
            negative = self.grouped_inputs[negative_item_idx]
            negative_label = self.grouped_labels[negative_item_idx]
            assert anchor_label != negative_label
        else:
            negative = []
        return [anchor, positive, negative], self.grouped_labels[item]

    def __len__(self):
        return len(self.grouped_examples)


    def get_group_info(self, features: List[SentenceLabelFeature]):
        """
        对所有 examples 按照 label 分组，并记录分组的信息
        返回
        # 记录分组后的 examples
        self.grouped_examples = []
        # 记录分组后的 右边届
        self.groups_right_border = []
        # 记录分组后的 example index
        self.idxs = []

        Assumes only one sentence per InputExample and labels as integers from 0 to max_num_labels
        and should be used in combination with dataset_reader.LabelSentenceReader.

        Labels with only one example are ignored.

        :param examples:
            the input examples for the training

        """
        # 新建一个 label：ex_index 对应的字典
        label_sent_mapping = {}
        logger.info("根据 label 对训练数据进行分组")
        # Group examples and labels
        # Add examples with the same label to the same dict
        for ex_index, feature in enumerate(tqdm(features, desc="Group feature")):
            if feature.label in label_sent_mapping:
                label_sent_mapping[feature.label].append(ex_index)
            else:
                label_sent_mapping[feature.label] = [ex_index]

        # Group sentences, such that sentences with the same label
        # are besides each other. Only take labels with at least 2 examples
        distinct_labels = list(label_sent_mapping.keys())
        for i in range(len(distinct_labels)):
            label = distinct_labels[i]
            # 只提取 同组数量大于 1 的 数据
            ex_indexes = label_sent_mapping[label]
            if len(ex_indexes) > 1:
                # 获取 label 对应的 grouped_examples
                grouped_features = [features[ex_index] for ex_index in ex_indexes]
                # 记录分组后的 examples
                self.grouped_features.extend(grouped_features)
                # 同一个组的右边届
                self.groups_right_border.append(
                    len(self.grouped_features))  # At which position does this label group / bucket end?
                self.num_labels += 1
        # 记录分组后的 example index
        self.grouped_indexes = range(len(self.grouped_features))

        logger.info("Num sentences: %d" % (len(self.grouped_features)))
        logger.info("Number of labels with >1 examples: {}".format(len(distinct_labels)))


    def get_feature(self, item):
        """
        @author:wangfc27441
        @desc: 输入 item(index),输出 index 对应 的 feature
        @version：
        @time:2020/10/30 9:40

        """
        # 从 grouped_examples 中获取 example
        feature = self.grouped_examples[item]

        return feature.text,feature.label


    def get_border(self, item):
        """
        @author:wangfc27441
        @desc: 获取 item（index） 的左右边界
        @version：
        @time:2020/10/30 9:41

        """
        # Check start and end position for this label in our list of grouped sentences
        group_idx = bisect.bisect_right(self.groups_right_border, item)
        left_border = 0 if group_idx == 0 else self.groups_right_border[group_idx - 1]
        right_border = self.groups_right_border[group_idx]
        return left_border, right_border



class SentenceLabelDatasetV2(IterableDataset)  :
    """
    主要代码源自：sentence-transformers v0.4.1
    输入： features
    输出： Dataset (IterableDataset 形式的，因为这样可以自己定义抽样的顺序)

    生成 Dataloader：
    examples -> features ->  dataset -> dataloader

    调用 Dataloader：
    dataloader 是 Iterable 对象， 调用 for 循环的时候，相当于调用 __iter__() 函数，生成 _BaseDataLoaderIter
    在做 for 循环的时候，相当于 调用 next() 函数，调用 _BaseDataLoaderIter 对象的 __next__() 函数


    This dataset can be used for some specific Triplet Losses like BATCH_HARD_TRIPLET_LOSS which requires
    multiple examples with the same label in a batch.
    It draws n consecutive, random and unique samples from one label at a time. This is repeated for each label.
    Labels with fewer than n unique samples are ignored.
    This also applied to drawing without replacement, once less than n samples remain for a label, it is skipped.
    This *DOES NOT* check if there are more labels than the batch is large or if the batch size is divisible
    by the samples drawn per label.
    """
    def __init__(self, examples: List[SentenceLabelFeature], samples_per_label: int = 4, with_replacement: bool = False,
                 least_num_per_label:int=2,
                 data_type:str='train'):
        """
        Creates a LabelSampler for a SentenceLabelDataset.
        :param examples:
            a list with InputExamples
        :param samples_per_label:
            the number of consecutive, random and unique samples drawn per label. Batch size should be a multiple of samples_per_label
        :param with_replacement:
            if this is True, then each sample is drawn at most once (depending on the total number of samples per label).
            if this is False, then one sample can be drawn in multiple draws, but still not multiple times in the same
            drawing.
        训练的时候，我们使用 hard_triplet_training的方法，因此对训练数据的抽样是需要特别设计的,返回 IterableDataset
        """
        super().__init__()
        self.data_type = data_type
        # 增加 每个label最小的数量
        self.least_num_per_label = least_num_per_label
        # 改变该参数为抽样的时候取值个数
        self.samples_per_label =  samples_per_label

        #Group examples by label
        label2ex = {}
        for example in examples:
            if example.label not in label2ex:
                label2ex[example.label] = []
            label2ex[example.label].append(example)

        #Include only labels with at least 2 examples
        self.grouped_inputs = []
        self.groups_right_border = []
        num_labels = 0

        for label, label_examples in label2ex.items():
            # 每个label最小的数量
            if len(label_examples) >= self.least_num_per_label:
                self.grouped_inputs.extend(label_examples)
                self.groups_right_border.append(len(self.grouped_inputs))  # At which position does this label group / bucket end?
                num_labels += 1

        self.label_range = np.arange(num_labels)
        self.with_replacement = with_replacement
        np.random.shuffle(self.label_range)
        logger.info("SentenceLabelDataset: {} examples, from which {} examples could be used (those labels appeared at least {} times). {} different labels found."
                    .format(len(examples), len(self.grouped_inputs), self.least_num_per_label, num_labels))


    def __iter__(self):
        """
        @author:wangfc
        @desc:
        可迭代样式的数据集是 IterableDataset 子类的实例，该子类实现了__iter__()协议，并表示数据样本上的可迭代。 这种类型的数据集特别适用于随机读取价格昂贵甚至不大可能，并且批处理大小取决于所获取数据的情况。
            例如，这种数据集称为iter(dataset)时，可以返回从数据库，远程服务器甚至实时生成的日志中读取的数据流。

        对于迭代式数据集，数据加载顺序完全由用户定义的迭代器控制。 这样可以更轻松地实现块读取和动态批次大小的实现(例如，通过每次生成一个批次的样本）。
        @version：
        @time:2021/3/8 14:12

        Parameters
        ----------

        Returns
        -------
        """
        label_idx = 0
        count = 0
        already_seen = {}
        while count < len(self.grouped_inputs):
            # 选取 label
            label = self.label_range[label_idx]
            if label not in already_seen:
                already_seen[label] = set()
            # 确定该 label 对应的 左右边界
            left_border = 0 if label == 0 else self.groups_right_border[label-1]
            right_border = self.groups_right_border[label]

            # 是否可以重复
            if self.with_replacement:
                selection = np.arange(left_border, right_border)
            else:
                selection = [i for i in np.arange(left_border, right_border) if i not in already_seen[label]]

            if len(selection) > self.samples_per_label:
                # 当每个 label对应的数据多于 samples_per_label，进行随机抽样
                selection = np.random.choice(selection, self.samples_per_label, replace=False)

            # 对每个label 进行抽样
            for element_idx in selection:
                count += 1
                already_seen[label].add(element_idx)
                # 使用 generator 来生成 sample, 我们需要输出一个字典格式
                yield self.grouped_inputs[element_idx].to_dict()

            # 当 上一个 generator 完成的时候，进行下一个
            label_idx += 1
            if label_idx >= len(self.label_range):
                label_idx = 0
                already_seen = {}
                np.random.shuffle(self.label_range)

    def __len__(self):
        return len(self.grouped_inputs)



class SentenceLabelDataset(Dataset):
    def __init__(self,examples: List[SentenceLabelFeature],data_type='test'):
        # 测试的时候，我们只需要一般的 dataset 就可以了
        super(SentenceLabelDataset, self).__init__()
        self.features = examples

    def __getitem__(self, item):
        return self.features[item].to_dict()

    def __len__(self):
        return len(self.features)

