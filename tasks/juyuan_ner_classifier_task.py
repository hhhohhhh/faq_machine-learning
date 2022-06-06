#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file:
@version:
@desc:
@time: 2021/12/28 9:20

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/28 9:20   wangfc      1.0         None
"""
from typing import Tuple, Dict, Text

import os
import pandas as pd

from data_process.data_processing import DataSpliter, is_single_label
from data_process.data_processor import RawDataProcessor
from data_process.dataset.text_classifier_dataset import TextClassifierProcessor
from tasks.text_classifier_task import TextClassifierTask, TransformerClassifierTask
# 事件类型定义版本信息
from data_process.dataset.juyuan_ner_classifier_dataset import JuyuanEventDefinitionDataV2, \
    JuyuanEventDefinitionDataV3, JuyuanRawDataV2, JuyuanEventDefinitionDifference, EVENT_LABEL_CODE_COLUMN, LINK_COLUMN, \
    FORTH_EVENT_LABEL_COLUMN, JuyuanNerClassifierPipelineProcessorV2, THIRD_EVENT_LABEL_COLUMN, DEFAULT_THIRD_EVENT_LABEL

import logging

from utils.io import dataframe_to_file, load_json, dump_json

logger = logging.getLogger(__name__)


class JuyuanNerClassifierTask(TransformerClassifierTask):
    def __init__(self, dataset='juyuan-event-entity-data',
                 data_subdir='ner_classifier_pipeline_data_20211228',
                 definition_subdir='event_label_definition',
                 last_event_definition_file='恒生聚源融合事件-舆情事件优化-20200810.xlsx',
                 last_output_data_subdir='ner_classifier_pipeline_data_20211204',
                 last_output_raw_data_filename='juyuan_entity_event_classifier_data_20201214.json',
                 new_event_definition_file="舆情事件体系V2.2_20211125.xlsx",
                 new_raw_data_subdir="raw_data_20211027",
                 new_raw_data_filename="语料new_20211027.xlsx",
                 transformed_raw_data_filename='juyuan_entity_event_classifier_data.json',
                 use_third_event_label = True,
                 filter_source= None,
                 *args, **kwargs):
        super(JuyuanNerClassifierTask, self).__init__(dataset=dataset,
                                                      data_subdir=data_subdir,
                                                      *args, **kwargs)
        self.definition_subdir = definition_subdir
        self.last_event_definition_file = last_event_definition_file
        self.last_output_data_subdir = last_output_data_subdir
        self.last_output_raw_data_filename = last_output_raw_data_filename

        self.new_event_definition_file = new_event_definition_file
        self.new_raw_data_subdir = new_raw_data_subdir
        self.new_raw_data_filename = new_raw_data_filename
        self.transformed_raw_data_filename = transformed_raw_data_filename
        # 聚源提供的原始数据是四级标签，现在模型需要训练三级事件标签
        self.use_third_event_label = use_third_event_label

        self.juyuan_event_definition_difference = self._get_definition_data()


    def _get_definition_data(self) -> JuyuanEventDefinitionDifference:
        """
        读取聚源事件定义体系版本变换情况
        """
        # 读取上一个版本的聚源事件定义体系
        last_event_definition_data = JuyuanEventDefinitionDataV3(
            event_definition_file=self.last_event_definition_file,
            output_data_subdir=self.last_output_data_subdir)
        # 读取最新版本的聚源事件体系
        new_event_definition_data = JuyuanEventDefinitionDataV3(
            event_definition_file=self.new_event_definition_file,
            output_data_subdir=self.data_subdir,
            if_get_forth_event_to_info_dict=True
        )

        juyuan_event_definition_difference = JuyuanEventDefinitionDifference(last_event_definition_data,
                                                                             new_event_definition_data)

        return juyuan_event_definition_difference

    def _prepare_data(self):
        """
        1. 准确训练和评估的原始数据
        2. 数据分析和统计
        3. 对数据进行分割为 train, test 数据集
        """
        # # 1. 准确训练和评估的原始数据
        # transformed_raw_data = self._prepare_raw_data()
        # # # 2. 数据分析和统计
        # # # 3. 对数据进行分割为 train, test 数据集
        # data_spliter = DataSpliter(data=transformed_raw_data, output_dir=self.data_dir,
        #                            train_size=0.9, test_size=0.1, split_strategy='stratify',
        #                            stratify_column=THIRD_EVENT_LABEL_COLUMN,
        #                            )
        #
        # raw_train_data, raw_dev_data, raw_test_data = data_spliter.get_split_data()

        raw_train_data =None

        # 从训练数据中筛选支持训练的类型，实验的的时候可能会过滤一些数据进行实验
        self.total_support_types_dict = self._get_support_types_dict(raw_train_data,
                                                               support_type_lowest_threshold=100,
                                                               data_dir=self.data_dir, )
        self.total_class_num = self.total_support_types_dict.__len__()

        # 创建 tokenizer
        self.tokenizer = self._create_tokenizer(tokenizer_type=self.tokenizer_type)

        # 使用 JuyuanNerClassifierPipelineProcessorV2   来加载数据创建 data_processor，进行数据预处理
        self.data_processor = JuyuanNerClassifierPipelineProcessorV2(
            debug=self.debug,
            data_dir=self.data_dir,
            output_dir=self.data_dir,
            output_model_dir = self.output_model_dir,
            # raw_train_data_filename=data_spliter._train_filename,
            # raw_dev_data_filename=data_spliter._dev_filename,
            # raw_test_data_filename=data_spliter._test_filename,
            max_seq_length=self.max_seq_length,
            total_support_types_dict=self.total_support_types_dict,

            # support_types_dict=self.support_types_dict,
            tokenizer=self.tokenizer,
            use_third_event_label = self.use_third_event_label,
            train_batch_size=self.train_batch_size,
            batch_strategy=self.batch_strategy,
            train_data_lowest_threshold = self.train_data_lowest_threshold,
            train_data_highest_threshold = self.train_data_highest_threshold,
            filter_source= self.filter_source,
            filter_single_label_examples = self.filter_single_label_examples

        )
        # 获取 train_dataset 输入模型训练中
        self.train_dataset, self.eval_dataset, self.test_dataset = self._build_model_data(
            )
        # 设置 model_support_types_dict
        self.model_support_types_dict = self.data_processor.model_support_types_dict

    def _prepare_raw_data(self):
        """
        准确训练和评估的原始数据:
        1. 读取上一个版本的数据, 转换为当前定义体系的数据 并增加 三级事件类型
        2. 读取最新的数据，转换为当前定义体系的数据 并增加 三级事件类型
        3. 合并数据

        """
        transformed_raw_data_path = os.path.join(self.data_dir, self.transformed_raw_data_filename)
        if os.path.exists(transformed_raw_data_path):
            transformed_raw_data = dataframe_to_file(path=transformed_raw_data_path, mode='r')
        else:
            # 1. 读取上一个版本的数据并进行转换
            last_raw_data_object = JuyuanRawDataV2(
                juyuan_event_definition_difference=self.juyuan_event_definition_difference,
                output_data_subdir=self.last_output_data_subdir,
                raw_data_json_filename=self.last_output_raw_data_filename,
                if_transform_data_to_single_entity_format=False,
                filter_sources="juyuan-human-labeled",
                entity_end_position_exclude=False,
            )

            # 2. 读取新增的原始数据并进行转换
            new_raw_data_object = JuyuanRawDataV2(
                juyuan_event_definition_difference=self.juyuan_event_definition_difference,
                raw_data_subdir=self.new_raw_data_subdir,
                raw_data_filename=self.new_raw_data_filename,
                output_data_subdir=self.data_subdir,
            )

            # 3. 合并数据
            last_transformed_raw_data = last_raw_data_object.transformed_raw_data
            new_transformed_raw_data = new_raw_data_object.transformed_raw_data

            transformed_raw_data = pd.concat([last_transformed_raw_data, new_transformed_raw_data])
            # 去除不重要的列
            transformed_raw_data.drop(axis=1, labels=[EVENT_LABEL_CODE_COLUMN, LINK_COLUMN], inplace=True)
            dataframe_to_file(path=transformed_raw_data_path, data=transformed_raw_data, mode='w')
        return transformed_raw_data

    def _get_support_types_dict(self, raw_train_data: pd.DataFrame,
                                label_column=THIRD_EVENT_LABEL_COLUMN, default_label=DEFAULT_THIRD_EVENT_LABEL,
                                support_type_lowest_threshold=100,
                                data_dir=None,
                                support_types_dict_name='support_types_dict.json') -> Dict[Text, Dict[Text, int]]:
        support_types_dict_path = os.path.join(data_dir, support_types_dict_name)
        if os.path.exists(support_types_dict_path):
            support_types_dict = load_json(support_types_dict_path)

        else:
            # 选取 单标签的数据
            if_single_label = raw_train_data.loc[:, label_column].apply(lambda label: is_single_label(label))
            single_label_raw_train_data = raw_train_data.loc[if_single_label].copy()
            # 对单标签的数据进行统计

            label_count_series = single_label_raw_train_data.loc[:, label_column].apply(
                lambda label: label[0]).value_counts()
            support_label_count_series = label_count_series[label_count_series > support_type_lowest_threshold]

            if default_label in support_label_count_series.index:
                default_label_count = support_label_count_series.loc[default_label]
                # 是否可以这样 drop ？
                support_label_count_series.drop(default_label)
            else:
                default_label_count = 0

            support_types_dict = RawDataProcessor.create_support_types_dict_from_counter(label_counter=support_label_count_series,
                                                                                         default_label=default_label,
                                                                                         default_label_count=default_label_count)
            # label_index = 0
            # support_types_dict = {
            #     default_label: {SUPPORT_TYPE_INDEX: label_index, SUPPORT_TYPE_COUNT: default_label_count}}
            #
            # for label, label_count in support_label_count_series.items():
            #     label_index += 1
            #     support_types_dict.update({label: {SUPPORT_TYPE_INDEX: label_index, SUPPORT_TYPE_COUNT: label_count}})

            dump_json(json_object=support_types_dict, json_path=support_types_dict_path)
        return support_types_dict

