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
import os
import numpy as np
import tensorflow as tf

from bert4keras.snippets import sequence_padding
from data_process.data_example import InputExample
from data_process.data_generator import DataGenerator, TextClassifierDataGenerator
from tokenizations import tokenization
from data_process.data_processor import DataProcessor



class EventClassifierProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def __init__(self, use_spm=None, do_lower_case=True, support_types_dict=None, use_text_b=True, classifier='event',
                 ):
        super(EventClassifierProcessor, self).__init__(use_spm, do_lower_case)
        self.support_types_dict = support_types_dict
        self.use_text_b = use_text_b
        self.classifier = classifier

        tf.logging.info(
            "classifier={}, use_text_b={},support_types_dict={}".format(classifier, use_text_b, support_types_dict))

    def get_train_examples(self, data_dir, train_data_name='important_events.train'):
        """See base class."""
        # 原数据预处理：分割为训练集和测试集，获取多标签的数据
        train_path = os.path.join(data_dir, train_data_name)
        return self._create_multilabel_examples(train_path, 'train',
                                                use_text_b=self.use_text_b,
                                                classifier=self.classifier)

    def _convert_multilabel2list(self, label_ls):
        """
        @author:wangfc27441
        @time:  2019/10/21  15:50
        @desc: 将 label_ls 转化为 一维数组: [0,0,...,1,0,...1,0,...0]
        """
        multilabel_indicator = np.zeros(shape=len(self.support_types_dict), dtype=np.int8)
        for label in label_ls:
            # 获取数据的标签，并将其转换为 multilabel_indicator，其他不在 support_types_dict中的标签返回为None
            label_index = self.support_types_dict.get(label, None)
            if label_index is not None:
                multilabel_indicator[label_index] = 1
        return multilabel_indicator.tolist()

    def _create_multilabel_examples(self, data_path, set_type, data=None, web_predict=False, use_text_b=False,
                                    classifier='event'):
        """Creates examples for the training and dev sets."""
        examples = []
        # 从web端输入数据
        if web_predict:
            for index, sample in enumerate(data):
                guid = '%s-%s' % ('web_predict', index)
                title = sample.get('title', '')
                text = sample.get('text', '')
                if use_text_b:
                    # 把字符串变成unicode的字符串。这是为了兼容Python2和Python3，
                    # 因为Python3的str就是unicode，而Python2的str其实是bytearray，Python2却有一个专门的unicode类型。
                    text_a = tokenization.convert_to_unicode(title)
                    text_b = tokenization.convert_to_unicode(text)
                else:
                    # 设置预测样本的 text_a
                    text_a = tokenization.convert_to_unicode(title + text)
                    text_b = None
                # 设置预测样本的 label=[0,...,0]
                label = np.zeros(shape=len(self.support_types_dict), dtype=np.int8).tolist()
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        else:
            with open(data_path, mode='r', encoding='utf-8') as f:
                for (i, line) in enumerate(f.readlines()[:1000]):
                    guid = '%s-%s' % (set_type, i)
                    splited_line = line.strip().split('###')
                    if classifier == 'event':
                        if len(splited_line) < 4:
                            label_col = 0
                            title_col = 1
                            text_col = 2
                            id = None
                            article_type = None
                        else:
                            columns = ['id', 'label', 'title', 'text', 'article_type']
                            label_col = 1
                            title_col = 2
                            text_col = 3
                            id_col = 0
                            article_type_col = 4
                            id = splited_line[id_col].strip()
                            article_type = splited_line[article_type_col].strip()
                    elif classifier == 'industry' or classifier == 'codetest' or classifier == 'concept':
                        columns = ['id', 'label', 'title', 'text']
                        label_col = 1
                        title_col = 2
                        text_col = 3
                        id_col = 0
                        article_type = None
                        id = splited_line[id_col].strip()

                    if splited_line == columns:
                        pass
                    else:
                        label_ls_str = splited_line[label_col].strip()
                        label_ls = [label.strip() for label in label_ls_str.split('@')]
                        title = splited_line[title_col].strip()
                        text = splited_line[text_col].strip()
                        if text is None:
                            text = ''

                        if use_text_b:
                            text_a = tokenization.convert_to_unicode(title)
                            text_b = tokenization.convert_to_unicode(text)
                        else:
                            # 设置预测样本的 text_a:将 title和text合并在一起作为输入
                            text_a = tokenization.convert_to_unicode(title + text)
                            text_b = None
                        label_index_ls = self._convert_multilabel2list(label_ls)
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label_index_ls))
        return examples

    def get_dev_examples(self, data_dir, dev_data_name='important_events.dev'):
        """See base class."""
        dev_path = os.path.join(data_dir, dev_data_name)
        return self._create_multilabel_examples(dev_path, 'dev', use_text_b=self.use_text_b,
                                                classifier=self.classifier)

    def get_test_examples(self, data_dir, test_data_name='important_events.test'):
        """See base class."""
        dev_path = os.path.join(data_dir, test_data_name)
        return self._create_multilabel_examples(dev_path, 'test', use_text_b=self.use_text_b,
                                                classifier=self.classifier)

    # 定义web服务时候提取 examples 函数：
    def get_web_test_examples(self, data):
        """
        @author:wangfc27441
        @time:  2019/10/29  10:06
        @desc:  读取并转换web提取的数据
        version 2： 参数由 title,text 转换为 data = [{title: None,text:None}, ... ],从而可以读取多条数据
        :param title:
        :param text:
        :return:
        """
        return self._create_multilabel_examples(data_path=None, set_type=None, web_predict=True, data=data,
                                                use_text_b=self.use_text_b)

    def get_labels(self):
        """See base class."""
        return list(self.support_types_dict.keys())



