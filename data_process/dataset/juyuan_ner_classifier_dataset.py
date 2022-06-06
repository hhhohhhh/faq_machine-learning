#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/20 16:32 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/20 16:32   wangfc      1.0         None
"""
import collections
import json
import os
from functools import partial
from typing import List, Text, Dict, Tuple, Set, Union
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from data_process.data_analysis import TextClassifierDataAnalysis
from data_process.data_example import InputExample, INPUT_EXAMPLE_LABEL_KEY, GUID_KEY
from data_process.data_feature import _truncate_seq_pair, InputFeatures
from data_process.data_generator import KerasTextClassifierDataGenerator
from data_process.data_processor import SUPPORT_TYPE_COUNT, SUPPORT_TYPE_INDEX
from data_process.dataset.text_classifier_dataset import TextClassifierProcessor
from tokenizations.tokenization import FullTokenizer, ChineseBaiscTokenizerV2, get_subword_tokens
from utils.constants import TRAIN_DATATYPE
from utils.io import dump_json, load_json, dataframe_to_file
from utils.string import rfind
import logging

logger = logging.getLogger(__name__)

ORIGINAL_FORTH_EVENT_NAME_KEY = '原有事件四级名称'

TITLE_COLUMN = 'title'
CONTENT_COLUMN = 'content'
ID_COLUMN = 'id'
ENTITY_KEY = "entity"
ENTITY_INDEX_KEY = 'entity_index'
ENTITY_TAG_KEY = "entity_tag"
ENTITY_START_POSITION_KEY = "start_position"
ENTITY_END_POSITION_KEY = "end_position"
FORTH_EVENT_LABEL_COLUMN = "forth_event_label"
# 新增三级事件类型的列
THIRD_EVENT_LABEL_COLUMN = 'third_event_label'

ENTITY_ARGUMENT_KEY = "entity_argument"
ENTITY_LABEL_ARGUMENT_KEY = "entity_label_argument"
RELEASE_DATE_COLUMN = "date"

# 读取 新增数据的 excel
EVENT_LABEL_CODE_COLUMN = 'label_code'
STANDARD_ENTITY_COLUMN = 'standard_entity'
LINK_COLUMN = 'link'
SOURCE_COLUMN = 'source'

ENTITY_CONTEXT_KEY = 'entity_context'

DEFAULT_THIRD_EVENT_LABEL = '其他'


# SUPPORT_TYPES_VALUE_NAMEDTUPLE = collections.namedtuple("support_types_value",field_names=["index",'count'])

TITLE_CONTENT_JOINED_CHAR = '\n'
SENTENCE_END_SEGMENT = '。'

# 石智友提供的数据
COLUMN_MAPPING_V2 = {'资讯id': ID_COLUMN,
                     '标题': TITLE_COLUMN,
                     '正文': CONTENT_COLUMN,
                     '四级事件分类': FORTH_EVENT_LABEL_COLUMN,
                     '四级事件code': EVENT_LABEL_CODE_COLUMN,
                     '标准化主体名称': STANDARD_ENTITY_COLUMN,
                     '事件依据': ENTITY_LABEL_ARGUMENT_KEY,
                     '信息发布日期': RELEASE_DATE_COLUMN,
                     '链接': LINK_COLUMN,
                     '实体的开始位置': ENTITY_START_POSITION_KEY,
                     '实体的结束位置': ENTITY_END_POSITION_KEY,
                     '原文主体名称': ENTITY_KEY}


class JuyuanEventDefinitionDataV2():
    """
    @time:  2021/11/8 10:44
    @author:wangfc
    @version:
    @description:

    @params:
    @return:
    """

    def __init__(self, corpus='corpus', dataset='juyuan-event-entity-data',
                 definition_subdir='event_label_definition',
                 output_data_subdir='ner_classifier_pipeline_data_20211108',
                 event_definition_file='恒生聚源融合事件-舆情事件优化-20200810.xlsx',
                 event_definition_compare_sheet='变更前后对比',
                 event_definition_sheet='变更后版本',
                 new_old_mapping_file="舆情事件编码映射-20201127.xlsx",
                 event_code_sheet='新版',
                 third_event_column='三级分类',
                 forth_event_column='四级分类',
                 event_definition_info_dict_filename='event_definition_info_dict.json',
                 ori_to_new_label_mapping_dict_name="ori_to_new_label_mapping_dict.json"):
        self._corpus = corpus
        self._dataset = dataset
        self._definition_subdir = definition_subdir
        self._output_data_subdir = output_data_subdir

        self._event_definition_file = event_definition_file
        self._event_definition_compare_sheet = event_definition_compare_sheet
        self._event_definition_sheet = event_definition_sheet
        self._new_old_mapping_file = new_old_mapping_file
        self._event_code_sheet = event_code_sheet
        self._third_event_column = third_event_column
        self._forth_event_column = forth_event_column

        self._event_definition_info_dict_filename = event_definition_info_dict_filename
        self._ori_to_new_label_mapping_dict_name = ori_to_new_label_mapping_dict_name

        # 读取所有的四级分类
        self._definition_dir = os.path.join(corpus, dataset, definition_subdir)
        self._output_data_dir = os.path.join(corpus, dataset, output_data_subdir)
        self._event_definition_path = os.path.join(self._definition_dir, event_definition_file)
        # self.ori_to_new_label_mapping_dict_path = os.path.join(self.definition_dir,ori_to_new_label_mapping_dict_name)
        self._event_definition_info_dict_path = os.path.join(self._output_data_dir, event_definition_info_dict_filename)

        # 事件定义体系的表格
        self.event_definition_df = None
        # 所有三级事件类型
        self.event_definition_set_3rd: Set[Text] = None
        # 所有四级事件类型
        self.event_definition_set_4th: Set[Text] = None

        self.get_event_definition_info_dict()
        # ori_to_new_label_mapping_dict = load_json(self.ori_to_new_label_mapping_dict_path)

    def get_event_definition_info_dict(self):
        self.event_definition_df = self._get_event_definition_table(sheet_name=self._event_definition_sheet)
        self.event_definition_set_3rd, self.event_definition_set_4th = self._get_event_definition_set()

    def _get_event_definition_table(self, sheet_name=None) -> pd.DataFrame:
        event_definition_df = pd.read_excel(self._event_definition_path, sheet_name=sheet_name,
                                            dtype=str, na_filter=False)
        columns = event_definition_df.columns.tolist()
        # event_definition_set = set(event_definition_df.loc[:, self.forth_event_column].dropna().tolist())
        logger.info("读取事件类型数据 shape = {},columns={} from {}".format(event_definition_df.shape, columns,
                                                                    self._event_definition_path))
        return event_definition_df

    def _get_event_definition_set(self) -> Tuple[Set[Text], Set[Text]]:
        event_definition_set_3rd = set(self.event_definition_df.loc[:, self._third_event_column].dropna().tolist())
        event_definition_set_4th = set(self.event_definition_df.loc[:, self._forth_event_column].dropna().tolist())
        logger.info(
            "三级事件类型共{}个，四级事件类型共{}个".format(event_definition_set_3rd.__len__(), event_definition_set_4th.__len__()))
        return event_definition_set_3rd, event_definition_set_4th

    def _get_other_in_event_name(self, event_name):
        if '其他' in event_name:
            return event_name

    def __repr__(self):
        return f"聚源事件定义体系:三级事件类型共{self.event_definition_set_3rd.__len__()}," \
               f"四级事件类型共{self.event_definition_set_4th.__len__()}"


class JuyuanEventDefinitionDataV3(JuyuanEventDefinitionDataV2):
    """

    """

    def __init__(self, event_definition_file='舆情事件体系V2.2_20211125.xlsx',
                 event_definition_compare_sheet='变更前后对比0928（含V2.1）',
                 event_definition_sheet='变更后版本',
                 event_code_sheet='编码0928',
                 forth_event_to_info_dict_filename='forth_event_to_info_dict.json',
                 original_forth_event_name_key=ORIGINAL_FORTH_EVENT_NAME_KEY,
                 if_get_forth_event_to_info_dict=False,
                 **kwargs,
                 ):
        super(JuyuanEventDefinitionDataV3, self).__init__(event_definition_file=event_definition_file,
                                                          event_definition_compare_sheet=event_definition_compare_sheet,
                                                          event_definition_sheet=event_definition_sheet,
                                                          event_code_sheet=event_code_sheet,
                                                          **kwargs)
        # 所有本版本支持的四级类型信息，包括编码等
        self.forth_event_to_info_dict: Dict[Text, Dict[Text, Text]] = None
        # 原有版本和当前版本修改的事件类型
        self.original_to_new_forth_event_dict: Dict[Text, Text] = None

        if if_get_forth_event_to_info_dict:
            # 新增 forth_event_to_info_dict 参数，保存 四级事件对应的定义信息
            self._forth_event_to_info_dict_filename = forth_event_to_info_dict_filename
            # 新增 original_forth_event_name_key，用于记录 forth_event_to_info_dict 中原有意图的名称
            self._original_forth_event_name_key = original_forth_event_name_key

            self._forth_event_to_info_dict_path = os.path.join(self._output_data_dir,
                                                               self._forth_event_to_info_dict_filename)

            # 获取 事件编码的表格：编码0928
            self.event_code_df = self._get_event_definition_table(sheet_name=self._event_code_sheet)
            self._modified_forth_events = self._get_modified_event_names()
            self._deleted_forth_events = self._get_modified_event_names(keyword="删除")
            self._added_forth_events = self._get_modified_event_names(keyword="新增")

            self.get_forth_event_to_info_dict()

    def get_special_event_names(self):
        def _get_key_word_in_event_name(event_name, key_word='其他'):
            if key_word in event_name:
                return event_name

        other_event_names = [event_name for event_name in self.event_definition_set_4th if
                             (_get_key_word_in_event_name(event_name))]
        other_event_names.__len__()

        key_word = '一般'
        general_event_name = [event_name for event_name in self.event_definition_set_4th if
                              (_get_key_word_in_event_name(event_name, key_word=key_word))]
        # general_event_name.__len__()
        print(general_event_name)

        key_word = '可能'
        maybe_event_names = [event_name for event_name in self.event_definition_set_4th if
                             (_get_key_word_in_event_name(event_name, key_word=key_word))]
        print(maybe_event_names)

        key_word = '后续'
        maybe_event_names = [event_name for event_name in self.event_definition_set_4th if
                             (_get_key_word_in_event_name(event_name, key_word=key_word))]
        print(maybe_event_names)

    def get_forth_event_to_info_dict(self):
        """
        @time:  2021/11/8 13:58
        @author:wangfc
        @version:
        @description: 建立 四级事件的信息字典

        @params:
        @return:
        """
        if self.event_definition_df is None:
            self.event_definition_df = self._get_event_definition_table(sheet_name=self._event_definition_sheet)
        if self.event_definition_set_3rd is None or self.event_definition_set_4th is None:
            self.event_definition_set_3rd, self.event_definition_set_4th = self._get_event_definition_set()
        self.forth_event_to_info_dict = self._get_forth_event_to_info_dict()
        self.original_to_new_forth_event_dict = self._get_original_to_new_forth_event_name_dict()

    def _get_forth_event_to_info_dict(self, forth_event_column='事件四级名称',
                                      index_key='index',
                                      original_forth_event_name_key=ORIGINAL_FORTH_EVENT_NAME_KEY):
        """
        获取 当前版本的四级事件对应的信息
        """
        if os.path.exists(self._forth_event_to_info_dict_path):
            forth_event_to_info_dict = load_json(json_path=self._forth_event_to_info_dict_path)
        else:
            new_to_original_forth_event_name_dict = self._get_new_to_original_forth_event_name_dict()

            # 防止事件名称重复的情况
            forth_event_name_count = self.event_code_df.loc[:, forth_event_column].value_counts(ascending=False)
            assert forth_event_name_count.loc[forth_event_name_count > 1].shape[0] == 0

            forth_event_to_info_dict = {}
            for i in range(self.event_code_df.__len__()):
                row = self.event_code_df.iloc[i]
                row_as_dict = row.to_dict()

                row_as_dict.update({index_key: i})
                forth_event_name = row.loc[forth_event_column].strip()
                if forth_event_name in new_to_original_forth_event_name_dict:
                    row_as_dict.update({index_key: i,
                                        original_forth_event_name_key: new_to_original_forth_event_name_dict[
                                            forth_event_name]})

                else:
                    row_as_dict.update({index_key: i})

                forth_event_to_info_dict.update({forth_event_name: row_as_dict})

            assert forth_event_to_info_dict.__len__() == self.event_code_df.__len__()
            dump_json(forth_event_to_info_dict, json_path=self._forth_event_to_info_dict_path)
        return forth_event_to_info_dict

    def _get_new_to_original_forth_event_name_dict(self):
        """
        获取上一版版本vs 当前版本四级事件名称变换的对应字典
        """
        # 读取 变更前后对比0928（含V2.1）
        event_definition_compare_df = self._get_event_definition_table(sheet_name=self._event_definition_compare_sheet)
        new_forth_event_column = f"{self._forth_event_column}.1"
        modified_event_names_compare_df = event_definition_compare_df.loc[event_definition_compare_df.loc[:, new_forth_event_column].isin(
            self._modified_forth_events)] \
                                              .loc[:, [self._forth_event_column, new_forth_event_column]]
        assert modified_event_names_compare_df.__len__() == self._modified_forth_events.__len__()
        # 获取四级事件名称变换的对应字典
        new_to_original_forth_event_name_dict = {}
        for index in modified_event_names_compare_df.index:
            new_forth_event_name = modified_event_names_compare_df.loc[index].loc[new_forth_event_column]
            forth_event_name = modified_event_names_compare_df.loc[index].loc[self._forth_event_column]
            new_to_original_forth_event_name_dict.update({new_forth_event_name: forth_event_name})
        return new_to_original_forth_event_name_dict

    def _get_modified_event_names(self, keyword='四级事件名称修改',
                                  event_column='事件四级名称',
                                  modify_tag_column="修改", ) -> List[Text]:
        modify_tag_not_na_event_code_df = self.event_code_df[~self.event_code_df.loc[:, modify_tag_column].isna()]
        has_modified = modify_tag_not_na_event_code_df.loc[:, modify_tag_column].str.contains(keyword)
        modified_event_code_df = modify_tag_not_na_event_code_df.loc[has_modified]
        modified_event_names = modified_event_code_df.loc[:, event_column].values.tolist()
        return modified_event_names

    def _get_original_to_new_forth_event_name_dict(self):
        original_to_new_forth_event_name_dict = {}
        for forth_event_name, event_info in self.forth_event_to_info_dict.items():
            original_forth_event_name = event_info.get(self._original_forth_event_name_key)
            if original_forth_event_name:
                original_to_new_forth_event_name_dict.update({original_forth_event_name: forth_event_name})
        return original_to_new_forth_event_name_dict

    def _get_forth_event_label_to_third_event_label(self, third_event_label_key="事件三级名称") -> Dict[Text, Text]:
        forth_event_label_to_third_event_label = {forth_event_label: event_info[third_event_label_key]
                                                  for forth_event_label, event_info in
                                                  self.forth_event_to_info_dict.items()}
        return forth_event_label_to_third_event_label


class JuyuanEventDefinitionDifference():
    """
    对比前后两个版本的差别
    """

    def __init__(self, last_event_definition_data: JuyuanEventDefinitionDataV3,
                 new_event_definition_data: JuyuanEventDefinitionDataV3):
        self.last_event_definition_data = last_event_definition_data
        self.new_event_definition_data = new_event_definition_data
        self._get_event_changed_info()

    def _get_event_changed_info(self):
        """
        获取三级或者四级事件定义的变化情况
        四级事件变换情况：
        1）修改名称： 对原有数据的四级事件名称进行变换
        2）去除 : 保留原有的数据，但是训练的时候需要进行过滤不参加训练
        3）新增:  新增的数据
        """
        deleted_event_definition_set_3rd = self.last_event_definition_data.event_definition_set_3rd. \
            difference(self.new_event_definition_data.event_definition_set_3rd)
        added_event_definition_set_3rd = self.new_event_definition_data.event_definition_set_3rd. \
            difference(self.last_event_definition_data.event_definition_set_3rd)

        # 修改名称的原有四级事件类型
        modified_original_forth_event_set = set(self.new_event_definition_data.original_to_new_forth_event_dict.keys())
        #  修改名称的新四级事件类型
        modified_forth_event_set = set(self.new_event_definition_data.original_to_new_forth_event_dict.values())

        # 删除的四级事件类型
        deleted_event_definition_set_4th = self.last_event_definition_data.event_definition_set_4th. \
            difference(self.new_event_definition_data.event_definition_set_4th) \
            .difference(modified_original_forth_event_set)

        # for event_name in self.new_event_definition_data.event_definition_set_4th:
        #     if "聚源" in event_name:
        #         print(event_name)

        # 新增的事件类型
        added_event_definition_set_4th = self.new_event_definition_data.event_definition_set_4th. \
            difference(self.last_event_definition_data.event_definition_set_4th) \
            .difference(modified_forth_event_set)

        self._deleted_event_definition_set_3rd = deleted_event_definition_set_3rd
        self._added_event_definition_set_3rd = added_event_definition_set_3rd

        self._modified_original_forth_event_set_4th = modified_original_forth_event_set
        self._deleted_event_definition_set_4th = deleted_event_definition_set_4th
        self._added_event_definition_set_4th = added_event_definition_set_4th

        assert self._deleted_event_definition_set_4th == set(self.new_event_definition_data._deleted_forth_events)
        assert self._added_event_definition_set_4th == set(self.new_event_definition_data._added_forth_events)

        logger.info(
            f"三级事件定义体系对比,去除原三级事件共 {deleted_event_definition_set_3rd.__len__()} 个:{deleted_event_definition_set_3rd}。"
            f"新增三级事件共 {added_event_definition_set_3rd.__len__()} 个:{added_event_definition_set_3rd}")
        logger.info(
            f"四级事件定义体系对比,去除原四级事件共 {deleted_event_definition_set_4th.__len__()} 个:{deleted_event_definition_set_4th}。"
            f"新增四级事件共 {added_event_definition_set_4th.__len__()} 个:{added_event_definition_set_4th},"
            f"修改的四级事件共 {self.new_event_definition_data.original_to_new_forth_event_dict.__len__()} 个:"
            f"{self.new_event_definition_data.original_to_new_forth_event_dict}")


class EntityLabelInfo():
    entity_key = ENTITY_KEY
    entity_tag_key = ENTITY_TAG_KEY

    def __init__(self, entity, entity_tag='nt',
                 start_position: int = -1, end_position: int = -1,
                 label: List[Text] = None,
                 entity_argument: Text = None,
                 entity_label_argument: Text = None,
                 *args, **kwargs):
        """
        每个实体的标注格式
        """
        self.entity = entity
        self.entity_tag = entity_tag
        self.start_position = start_position
        self.end_position = end_position
        self.label = label
        self.entity_argument = entity_argument
        self.entity_label_argument = entity_label_argument

    @property
    def entity_column(self):
        return

    def to_dict(self):
        entity_label_info_dict = {'entity': self.entity,
                                  'entity_tag': self.entity_tag,
                                  'start_position': self.start_position,
                                  'end_position': self.end_position,
                                  "label": self.label,
                                  'entity_argument': self.entity_argument,
                                  'entity_label_argument': self.entity_label_argument}
        return entity_label_info_dict


class NewsLabelInfo():
    def __init__(self, id: Text, title: Text, content: Text,
                 entity_label_info_ls: List[EntityLabelInfo], labels_ls: List[Text],
                 date: Text, source: Text, *args, **kwargs
                 ):
        """
        每条新闻的标注格式
        """
        self.id = id
        self.title = title
        self.content = content
        self.entity_label_info_ls = entity_label_info_ls
        self.labels_ls = sorted(list(labels_ls))
        self.date = date
        self.source = source

    def to_dict(self):
        news_label_info_dict = {'id': self.id,
                                'title': self.title,
                                'content': self.content,
                                'entity_label_info_ls': [e.to_dict() for e in self.entity_label_info_ls],
                                'labels_ls': self.labels_ls,
                                "date": self.date,
                                "source": self.source}
        return news_label_info_dict


class JuyuanRawDataV1():
    """
    樊孟军提供的原始数据是按照 一行 进行存储的 xlsx 格式存储的
    """
    pass


class JuyuanRawDataV2(JuyuanRawDataV1):
    def __init__(self,
                 juyuan_event_definition_difference: JuyuanEventDefinitionDifference,
                 corpus='corpus', dataset='juyuan-event-entity-data',
                 raw_data_subdir="raw_data_20201124",
                 raw_data_filename='语料new_20211027.xlsx',
                 output_data_subdir='ner_classifier_pipeline_data_20201204',
                 raw_data_json_filename=None,
                 if_transform_data_to_single_entity_format=False,
                 filter_sources=None,
                 entity_end_position_exclude=True
                 # output_raw_data_filename='juyuan_entity_event_classifier_data.json',

                 # last_raw_data_filename='juyuan_entity_event_classifier_data_20201214.json',
                 # new_data_subdir='ner_classifier_pipeline_new_data_20211027',
                 # new_data_filename='语料new_20211027.xlsx',
                 # output_data_subdir='ner_classifier_pipeline_data_20211109',

                 # all_data_filename='all.json',
                 # train_data_filename='train.json',
                 # dev_data_filename='dev.json',
                 # test_data_filename='test.json'
                 ):
        # 事件定义系统
        self._corpus = corpus
        self._dataset = dataset
        self._juyuan_event_definition_difference = juyuan_event_definition_difference

        self._raw_data_subdir = raw_data_subdir
        self._raw_data_filename = raw_data_filename

        self._raw_data_filename_stem = Path(self._raw_data_filename).stem
        if raw_data_json_filename is None:
            raw_data_json_filename = f"{self._raw_data_filename_stem}.json"

        self._raw_data_json_filename = raw_data_json_filename

        # 当前版本输出的目录
        self._output_data_subdir = output_data_subdir
        # 原始的数据名称 + 过滤后的数据名称 + 分割后的数据名称
        # self._output_raw_data_filename = output_raw_data_filename

        self._raw_data_dir = os.path.join(self._corpus, self._dataset, self._raw_data_subdir)
        self._raw_data_path = os.path.join(self._raw_data_dir, self._raw_data_filename)
        self._output_data_dir = os.path.join(self._corpus, self._dataset, self._output_data_subdir)
        # 输出为json格式的数据，方便快速读取
        self._raw_data_json_path = os.path.join(self._output_data_dir, self._raw_data_json_filename)

        # 是否讲以前转换为唯一id的数据转换为entity对应的数据
        self._if_transform_data_to_single_entity_format = if_transform_data_to_single_entity_format

        if isinstance(filter_sources, str):
            filter_sources = [filter_sources]
        self._filter_sources = filter_sources
        self._entity_end_position_exclude = entity_end_position_exclude

        self.raw_data = self.get_raw_data(
            if_transform_data_to_single_entity_format=self._if_transform_data_to_single_entity_format,
            filter_sources=self._filter_sources)
        self.transformed_raw_data = self._transform_raw_data()

        # 上一个版本的数据路径和原始数据名称:
        # 这版的原始数据是按照 id 号相同的数据进行存储的 json 格式存储的
        # self.last_output_subdir = last_output_subdir
        # self.last_raw_data_filename = last_raw_data_filename
        # # 新增数据的路径和名称
        # self.new_data_subdir = new_data_subdir
        # self.new_data_filename = new_data_filename

        # self._all_data_filename = all_data_filename
        # self._train_data_filename = train_data_filename
        # self._dev_data_filename = dev_data_filename
        # self._test_data_filename = test_data_filename

        # self.last_output_dir = os.path.join(self.corpus, self.dataset, self.last_output_subdir)
        # self.new_data_dir = os.path.join(self.corpus, self.dataset, self.new_data_subdir)

    def get_raw_data(self, data_version='v2',
                     if_transform_data_to_single_entity_format=False, filter_sources=None,if_change_label_column=True):
        if os.path.exists(self._raw_data_json_path):
            raw_data = dataframe_to_file(path=self._raw_data_json_path, mode='r')  # orient=None,lines=False
            if if_transform_data_to_single_entity_format:
                logger.info(f"对原始数据进行变换为 single entity的形式")
                raw_data = self._transform_data_to_single_entity_format(data=raw_data)
            if filter_sources:
                logger.info(f"对原始数据根据 source={filter_sources} 进行过滤")
                is_in_filter_sources = raw_data.loc[:, SOURCE_COLUMN].isin(filter_sources)
                raw_data = raw_data.loc[is_in_filter_sources]
            if if_change_label_column:
                columns = raw_data.columns.tolist()
                new_columns = columns_mapping(columns)
                raw_data.columns = new_columns
            entity_position_correct_raw_data = self._check_entity_positions(raw_data, self._entity_end_position_exclude)
            # 默认读取的时候是已经合并后的数据
            entity_label_combined_raw_data = entity_position_correct_raw_data

        else:
            raw_data = dataframe_to_file(path=self._raw_data_path, mode='r')
            # 获取对应的英文列名，并转换
            new_columns = self._get_new_columns(raw_data, data_version=data_version)
            raw_data.columns = new_columns

            if SOURCE_COLUMN not in raw_data.columns:
                raw_data.loc[:, SOURCE_COLUMN] = self._raw_data_filename

            # 去除 entity 为空的情况
            is_entity_na = raw_data.loc[:, ENTITY_KEY].isna()
            nona_entity_raw_data = raw_data.loc[~is_entity_na].copy()

            # 去除重复的情况
            nona_entity_raw_data.drop_duplicates(subset=[ID_COLUMN, ENTITY_KEY, FORTH_EVENT_LABEL_COLUMN,
                                                         ENTITY_START_POSITION_KEY, ENTITY_END_POSITION_KEY],
                                                 inplace=True)

            # 检查entity的位置
            entity_position_correct_raw_data = self._check_entity_positions(raw_data=nona_entity_raw_data)

            # TODO : 检验 多个实体的情况位置是否正确,新版数据中可能存在位置不准确的情况
            count_by_id_entity = entity_position_correct_raw_data.groupby([ID_COLUMN, ENTITY_KEY]).count() \
                .sort_values(by=TITLE_COLUMN, ascending=False)
            count_by_id_entity.loc[count_by_id_entity.loc[:, TITLE_COLUMN] > 2].loc[:, TITLE_COLUMN].sum()

            # 检查 时间范围
            min_release_date = entity_position_correct_raw_data.loc[:, RELEASE_DATE_COLUMN].min()
            max_release_date = entity_position_correct_raw_data.loc[:, RELEASE_DATE_COLUMN].max()

            # 如果 label 是 list 格式 或者 text 格式，合并 id ，entity，start_position 相同 entity label 信息
            entity_label_combined_raw_data = self._combine_entity_label(entity_position_correct_raw_data)
            dataframe_to_file(path=self._raw_data_json_path, data=entity_label_combined_raw_data, mode='w')

        return entity_label_combined_raw_data

    def _get_new_columns(self, raw_data, data_version='v2') -> List[Text]:
        if data_version == 'v2':
            columns = raw_data.columns.tolist()
            new_columns = [COLUMN_MAPPING_V2[column] for column in columns]
        return new_columns

    def _transform_data_to_unique_id_format(self, raw_data=None, data_path=None, id_column=ID_COLUMN,
                                            title_column=TITLE_COLUMN,
                                            content_column=CONTENT_COLUMN,
                                            event_label_column=FORTH_EVENT_LABEL_COLUMN,
                                            entity_label_argument_column=ENTITY_LABEL_ARGUMENT_KEY,
                                            entity_column=ENTITY_KEY,
                                            entity_start_position_column=ENTITY_START_POSITION_KEY,
                                            entity_end_position_column=ENTITY_END_POSITION_KEY,
                                            entity_argument_column=ENTITY_ARGUMENT_KEY,
                                            release_date_column=RELEASE_DATE_COLUMN
                                            ):
        """
        @author:wangfc27441    @desc:
        1. 之前保存为excel格式原始数据是按照 一个 entity为一行的形式保存的，现在合并 id 号相同的数据
        2. 根据映射表映射为新的 事件类型
        3. 合成字段 entity_label_info_ls：记录每个 entity 对应的信息
            labels_ls:  记录每个id 对应的 labels
        @version：
        @time:2020/12/4 11:26

        Parameters
        ----------

        Returns
        -------
        """

        # 读取原始数据
        if raw_data is None:
            raw_data = pd.read_excel(data_path, keep_default_na=False)
            raw_data.drop_duplicates(inplace=True)
            filename = os.path.split(data_path)[-1]
            columns = raw_data.columns.tolist()
            new_columns = [self.COLUMN_MAPPING[c] for c in columns]
            raw_data.columns = new_columns
            logger.info("raw_data.shape={},raw_data.columns={} from {}:\nevent_label_nums ={}"
                        .format(raw_data.shape, raw_data.columns.tolist(), data_path,
                                raw_data.loc[:, event_label_column].value_counts().shape))

        # 检验 entity position的正确性
        is_entity_position_correct = raw_data.apply(self._check_entity_positions, axis=1)
        entity_position_correct_ori_raw_data = raw_data[~is_entity_position_correct]

        # 验证实体必须在事件依据中
        # assert entity_position_correct_ori_raw_data.shape[0] == 0

        raw_data[title_column] = raw_data.title.apply(str.strip)
        raw_data[content_column] = raw_data.content.apply(str.strip)

        # 获取所有的id 号
        id_value_counts = raw_data.loc[:, id_column].value_counts()
        id_num = id_value_counts.__len__()
        new_rows = []

        # 按照 id 号 获取 entity-label 对应关系
        for i in range(id_num):
            id_index = id_value_counts.index[i]
            if i % 100 == 0:
                logger.info("开始处理第{}/{}个id：{}".format(i, id_num, id_index))
            id_data = raw_data[raw_data.loc[:, id_column] == id_index].copy()
            first_row = id_data.iloc[0]
            title = first_row[title_column]
            content = first_row[content]

            entity_label_info_ls = []
            # 记录 id 对应的数据
            id_labels_set = set()
            for i in range(id_data.__len__()):
                row = id_data.iloc[i]
                label = row[event_label_column]
                # 根据映射表进行映射新标签
                # new_label = new_label_mapping_fun(id_index, label, ori_to_new_label_mapping_dict)
                id_labels_set.add(label)

                entity = row[entity_column]
                start_position = row[entity_start_position_column]
                end_position = row[entity_end_position_column]
                entity_label_argument = row[entity_label_argument_column]
                entity_argument = row[entity_argument_column]

                date = row[release_date_column]
                entity_label_info = EntityLabelInfo(entity=entity,
                                                    start_position=start_position,
                                                    end_position=end_position,
                                                    label=[label],
                                                    entity_argument=entity_argument,
                                                    entity_label_argument=entity_label_argument)

                entity_label_info_ls.append(entity_label_info)
            news_label_info = NewsLabelInfo(id=id_index, title=title, content=content,
                                            entity_label_info_ls=entity_label_info_ls,
                                            labels_ls=sorted(list(id_labels_set)),
                                            date=date,
                                            source=filename)

            new_rows.append(news_label_info.to_dict())
        transformed_raw_data = pd.DataFrame(new_rows)
        return transformed_raw_data

    def _transform_data_to_single_entity_format(self, data: pd.DataFrame):
        """
        将合并为 唯一id 的数据变换为 按照 entity 格式的数据，方便进行原始数据转换等
        """
        new_data_ls = []
        for index in tqdm.trange(data.__len__(), desc="原始变换为single_entity格式"):
            news_label_info_dict = data.iloc[index].to_dict()
            # id_index = news_label_info_dict['id']
            # title = news_label_info_dict['title']
            # content = news_label_info_dict['content']

            entity_label_info_dict_ls = news_label_info_dict['entity_label_info_ls']
            # entity_label_info_ls = []
            for entity_label_info_dict in entity_label_info_dict_ls:
                # entity_label_info = EntityLabelInfo(**entity_label_info_dict)
                # entity_label_info_ls.append(entity_label_info)
                new_data_row = news_label_info_dict.copy()
                new_data_row.pop('entity_label_info_ls')
                new_data_row.pop('labels_ls')
                new_data_row.update(entity_label_info_dict)
                new_data_ls.append(new_data_row)
            # news_label_info = NewsLabelInfo(entity_label_info_ls=entity_label_info_ls,**news_label_info_dict)
        new_data = pd.DataFrame(new_data_ls)
        return new_data

    def _check_entity_positions(self, raw_data, entity_end_position_exclude=True):
        # is_entity_position_correct = raw_data.apply(self._check_entity_position_func, axis=1)
        # entity_position_not_correct_raw_data = raw_data[~is_entity_position_correct]
        # assert entity_position_not_correct_raw_data.shape[0] == 0
        check_entity_position_func = partial(self._check_entity_position_func,
                                             entity_end_position_exclude=entity_end_position_exclude)
        is_entity_position_correct = raw_data.apply(check_entity_position_func, axis=1)
        entity_position_not_correct_raw_data = raw_data[~is_entity_position_correct]

        if entity_position_not_correct_raw_data.shape[0] != 0:
            logger.warning(f"原始数据中存在 {entity_position_not_correct_raw_data.shape[0]} 条数据实体位置错误")

        entity_position_correct_raw_data = raw_data[is_entity_position_correct].copy()
        assert entity_position_correct_raw_data.shape[0] > 0

        if entity_end_position_exclude == False:
            # 统一修正为 exclude 模式,保持数据格式的一致性
            end_positions = entity_position_correct_raw_data.loc[:, ENTITY_END_POSITION_KEY].apply(
                lambda end_position: end_position + 1)
            entity_position_correct_raw_data.loc[:, ENTITY_END_POSITION_KEY] = end_positions
            check_entity_position_func = partial(self._check_entity_position_func, entity_end_position_exclude=True)
            is_entity_position_correct = entity_position_correct_raw_data.apply(check_entity_position_func, axis=1)
            entity_position_not_correct_raw_data = entity_position_correct_raw_data[~is_entity_position_correct]
            assert entity_position_not_correct_raw_data.shape[0] == 0
        return entity_position_correct_raw_data

    # 检验 entity position的正确性
    def _check_entity_position_func(self, row,
                                    entity_end_position_exclude=True,
                                    title_column=TITLE_COLUMN,
                                    content_column=CONTENT_COLUMN,
                                    title_content_joined_char=TITLE_CONTENT_JOINED_CHAR,
                                    entity_column=ENTITY_KEY,
                                    entity_start_position_column=ENTITY_START_POSITION_KEY,
                                    entity_end_position_column=ENTITY_END_POSITION_KEY
                                    ):
        # 原来的数据保存为excel格式后，开头会出现 特殊字符，需要使用 strip 去除后才能正确匹配
        title = row[title_column]
        content = row[content_column]
        entity = row[entity_column].lower().strip()
        entity_start_position = row[entity_start_position_column]
        entity_end_position = row[entity_end_position_column]

        is_entity_position_correct = JuyuanNerClassifierPipelineInputExampleV2. \
            _check_entity_position(title=title,
                                   content=content,
                                   entity=entity,
                                   title_content_joined_char=title_content_joined_char,
                                   entity_start_position=entity_start_position,
                                   entity_end_position=entity_end_position,
                                   entity_end_position_exclude=entity_end_position_exclude)
        return is_entity_position_correct

    def _combine_entity_label(self, raw_data: pd.DataFrame):
        """
        如果 label 是 list 格式 或者 text 格式，合并 id ，entity，start_position 相同 entity label 信息,后续可以生成多标签的数据
        """
        # 重置 index
        raw_data.reset_index(drop=True, inplace=True)
        # 对原始数据根据 [ID_COLUMN,ENTITY_KEY,ENTITY_START_POSITION_KEY] 进行分组
        grouped_data = raw_data.groupby(by=[ID_COLUMN, ENTITY_KEY, ENTITY_START_POSITION_KEY, ENTITY_END_POSITION_KEY])
        entity_label_combined_rows = []
        for group_key, group_indies in tqdm.tqdm(grouped_data.groups.items(), total=len(grouped_data),
                                                 desc='合并同一位置entity的label'):
            event_label_series = raw_data.loc[group_indies].loc[:, FORTH_EVENT_LABEL_COLUMN]
            event_labels_set = set()
            for event_label in event_label_series:
                if isinstance(event_label, str):
                    event_labels_set.add(event_label)
                elif isinstance(event_label, list):
                    event_labels_set = event_labels_set.union(set(event_label))
                else:
                    raise TypeError(f"event_label={event_label} type= {type(event_label)}不支持")
            event_labels = sorted(event_labels_set)
            data_row = raw_data.loc[group_indies[0]].copy()
            data_row.loc[FORTH_EVENT_LABEL_COLUMN] = event_labels
            entity_label_combined_rows.append(data_row)
        entity_label_combined_data = pd.DataFrame(entity_label_combined_rows)

        return entity_label_combined_data

    def _transform_raw_data(self):
        # 转换原始数据为 最新的事件定义体系数据
        raw_data = self.raw_data.copy()
        filter_raw_data = self._transform_new_event_definition_data(raw_data)
        # 增加三级事件标签
        transformed_raw_data = self._add_third_event_labels(filter_raw_data)
        return transformed_raw_data

    def _transform_new_event_definition_data(self, raw_data):
        """
        转换原始数据为 最新的事件定义体系数据

        四级事件变换情况：
        1）修改名称： 对原有数据的四级事件名称进行变换
        2）去除 : 保留原有的数据，但是训练的时候需要进行过滤不参加训练
        3）新增:  新增的数据

        对应的数据修改：
        1）修改名称
        2）去除

        """

        # 1）修改名称
        new_forth_event_labels = raw_data.loc[:, FORTH_EVENT_LABEL_COLUMN].apply(self._modified_forth_event_label)

        # 2）去除不属于的数据
        event_definition_set_4th = self._juyuan_event_definition_difference.new_event_definition_data.event_definition_set_4th
        droped_forth_event_labels = new_forth_event_labels \
            .apply(lambda event_labels: self._drop_event_label(event_labels, event_definition_set_4th))
        # 去除空的数据
        filter_raw_data = raw_data.copy()
        filter_raw_data.loc[:, FORTH_EVENT_LABEL_COLUMN] = droped_forth_event_labels
        filter_raw_data = filter_raw_data.loc[~filter_raw_data.loc[:, FORTH_EVENT_LABEL_COLUMN].isna()].copy()
        return filter_raw_data

    def _modified_forth_event_label(self, forth_event_labels: Union[Text, List[Text]]):
        original_to_new_forth_event_dict = self._juyuan_event_definition_difference.new_event_definition_data.original_to_new_forth_event_dict
        if isinstance(forth_event_labels, str):
            forth_event_labels = [forth_event_labels]
        new_forth_event_labels = []
        for forth_event_label in forth_event_labels:
            if forth_event_label in original_to_new_forth_event_dict:
                new_forth_event_label = original_to_new_forth_event_dict[forth_event_label]
                logger.debug(f"修正原有四级事件标签{forth_event_label} to {new_forth_event_label}")
            else:
                new_forth_event_label = forth_event_label
            new_forth_event_labels.append(new_forth_event_label)
        return new_forth_event_labels

    def _drop_event_label(self, event_labels: List[Text], event_definition_set_4th: Set[Text]) -> List[Text]:
        new_event_labels = [event_label for event_label in event_labels if event_label in event_definition_set_4th]
        if new_event_labels.__len__() > 0:
            return new_event_labels

    def _add_third_event_labels(self, raw_data):
        """
        增加三级事件的信息
        """
        forth_event_label_to_third_event_label = self._juyuan_event_definition_difference.new_event_definition_data. \
            _get_forth_event_label_to_third_event_label()

        third_event_labels = raw_data.loc[:, FORTH_EVENT_LABEL_COLUMN] \
            .apply(lambda forth_event_labels: self._add_third_event_label(forth_event_labels,
                                                                          forth_event_label_to_third_event_label))
        raw_data.loc[:, THIRD_EVENT_LABEL_COLUMN] = third_event_labels
        return raw_data

    def _add_third_event_label(self, forth_event_labels, forth_event_label_to_third_event_label) -> List[Text]:
        third_event_label_set = set()
        for forth_event_label in forth_event_labels:
            third_event_label = forth_event_label_to_third_event_label[forth_event_label]
            third_event_label_set.add(third_event_label)
        third_event_labels = list(third_event_label_set)
        return third_event_labels


class JuyuanNerClassifierPipelineInputExampleV2(InputExample):
    """A single training/test example for simple sequence classification."""
    """
    @author:wangfc27441
    @desc: 通过 先 ner 识别 再进行 Classifier 形成 一个 pipeline 的方式，
    优点：模型改造简单，沿用分类的模型，逐个对每个 entity 找到其位置对应的 窗口来进行分类
    缺点：对于多个 entity的时候，性能较差
    @version：
    @time:2020/5/26 18:01 
    """

    def __init__(self, guid,
                 id=None,
                 title="",
                 content="",
                 title_content_joined_char=TITLE_CONTENT_JOINED_CHAR,
                 sentence_end_segment=SENTENCE_END_SEGMENT,
                 entity_index=None,
                 entity=None, standard_entity=None,
                 entity_tag=None,
                 entity_argument=None, entity_label_argument=None,
                 start_position=None, end_position=None,
                 forth_event_label: List[Text] = None,
                 third_event_label: List[Text] = None,
                 source: Text = None,
                 # support_types_dict: Dict[Text, int] = None,
                 # use_third_event_label=True,
                 text_a: Text = None,
                 text_b: Text = None,
                 label: List[Text] = None,
                 label_id: List[int] = None,
                 * args,
                 **kwargs
                 ):
        """
        :param guid:
        :param ids:
        :param title:
        :param content:
        :param entity:
        :param entity_argument:
        :param entity_start_position:
        :param event_label:
        :param event_label_argument:
        """
        self.guid = guid
        self.id = id
        self.title = title
        self.content = content
        self.title_content_joined_char = title_content_joined_char
        self.sentence_end_segment = sentence_end_segment
        self.entity_index = entity_index
        self.entity = entity
        self.standard_entity = standard_entity  # 增加聚源接口提供 word
        self.entity_tag = entity_tag
        self.entity_argument = entity_argument
        self.start_position = int(start_position)
        self.end_position = int(end_position)
        # 分清楚三级四级事件标签
        self.forth_event_label = forth_event_label
        self.third_event_label = third_event_label
        self.entity_label_argument = entity_label_argument
        self.source = source
        # self.event_label_index = event_label_index
        # self.support_types_dict = support_types_dict
        self.args = args
        self.kwargs = kwargs

        self.entity_context = self._find_entity_context(title=title, content=content,
                                                        title_content_joined_char=title_content_joined_char,
                                                        entity=entity,
                                                        entity_start_position=start_position,
                                                        entity_end_position=end_position)

        # bert 的输入
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.label_id = label_id

        # if label_ids is None:
        #     if use_third_event_label:
        #         event_labels = third_event_label
        #     else:
        #         event_labels = label
        #     label_ids = JuyuanNerClassifierPipelineProcessorV2._convert_multilabel2list(event_labels,
        #                                                                                 support_types_dict)
        #     self.label_ids = label_ids

    def _find_entity_context(self, title, content, title_content_joined_char,
                             entity, entity_start_position, entity_end_position
                             ):
        """
        获取 entity 所在位置的 context
        """
        context = None
        is_entity_position_correct = self._check_entity_position(title=title, content=content, entity=entity,
                                                                 entity_start_position=entity_start_position,
                                                                 entity_end_position=entity_end_position,
                                                                 title_content_joined_char=title_content_joined_char)

        if is_entity_position_correct:
            # 如果entity出现在 title中,对 tokens_b保留原样，并再后面的处理过程中从前往后裁剪
            if entity_start_position < len(title):
                context = f"{title_content_joined_char}".join([title, content])
            # 如果 entity出现在 content 中，对 token_b 进行裁剪：
            else:
                # 找到 该entity所在语句，从上一句的 句号以后位置开始截取
                last_sentence_position_in_content = JuyuanNerClassifierPipelineProcessorV2 \
                    ._find_last_sentence_end_segment_position(title, content, entity_start_position,
                                                              title_content_joined_char)
                context = content[last_sentence_position_in_content + 1:]
                # 更新 entity_start_position
                new_entity_start_position = entity_start_position - len(title) - len(title_content_joined_char) - (
                        last_sentence_position_in_content + 1)
                new_entity_end_position = entity_end_position - len(title) - len(title_content_joined_char) - (
                        last_sentence_position_in_content + 1)
                # 验证新的 new_context中的entity位置
                entity_after_window = context[new_entity_start_position: new_entity_end_position].lower()

                if entity.lower() != entity_after_window:
                    logger.error(
                        "截取窗口后的entity:{} != entity_after_window:{}".format(entity.lower(), entity_after_window))
                    context = None
        return context

    @staticmethod
    def _check_entity_position(title, content, entity, entity_start_position, entity_end_position,
                               title_content_joined_char='\n',
                               entity_end_position_exclude=True, lower=True):
        context = title.strip() + title_content_joined_char + content.strip()
        if entity_end_position_exclude:
            entity_from_position = context[entity_start_position:entity_end_position]
        else:
            entity_from_position = context[entity_start_position:entity_end_position + 1]
        if lower:
            is_entity_position_correct = entity.lower() == entity_from_position.lower()
        else:
            is_entity_position_correct = entity == entity_from_position
        if is_entity_position_correct == False:
            logger.warning(f"entity ={entity} 位置错误！")
        return is_entity_position_correct

    def to_dict(self, with_model_input=False):
        output_dict = {GUID_KEY: self.guid,
                       ID_COLUMN: self.id,
                       TITLE_COLUMN: self.title,
                       CONTENT_COLUMN: self.content,
                       ENTITY_INDEX_KEY: self.entity_index,
                       ENTITY_KEY: self.entity,
                       STANDARD_ENTITY_COLUMN: self.standard_entity,  # 增加聚源接口提供 word
                       ENTITY_TAG_KEY: self.entity_tag,
                       ENTITY_ARGUMENT_KEY: self.entity_argument,
                       ENTITY_START_POSITION_KEY: self.start_position,
                       ENTITY_END_POSITION_KEY: self.end_position,
                       FORTH_EVENT_LABEL_COLUMN: self.forth_event_label,
                       THIRD_EVENT_LABEL_COLUMN: self.third_event_label,
                       INPUT_EXAMPLE_LABEL_KEY: self.label,
                       ENTITY_LABEL_ARGUMENT_KEY: self.entity_label_argument,
                       ENTITY_CONTEXT_KEY: self.entity_context,
                       SOURCE_COLUMN: self.source
                       }

        # if with_model_input:
        #     model_input = {'input_ids': self.input_ids,
        #                    'input_mask': self.input_mask,
        #                    'segment_ids': self.segment_ids,
        #                    'label_ids': self.label_ids}
        #     output_dict.update(model_input)

        return output_dict


class JuyuanNerClassifierPipelineProcessorV2(TextClassifierProcessor):
    """Processor for the XNLI data set."""
    """
    @author:wangfc27441
    @desc: 
    @version：
    @time:2020/5/26 17:59 

    """

    def __init__(self, debug=False, raw_train_data=None, raw_dev_data=None, raw_test_data=None,
                 use_third_event_label=True,
                 train_data_lowest_threshold =None,
                 train_data_highest_threshold=None,
                 filter_source=None,filter_single_label_examples=False,
                 *args, **kwargs):
        super(JuyuanNerClassifierPipelineProcessorV2, self).__init__(*args, **kwargs)
        self.debug = debug
        self.raw_train_data = raw_train_data
        self.raw_dev_data = raw_dev_data
        self.raw_test_data = raw_test_data
        self.use_third_event_label = use_third_event_label
        self.train_data_lowest_threshold = train_data_lowest_threshold
        self.train_data_highest_threshold = train_data_highest_threshold
        self.filter_source = filter_source
        self.filter_single_label_examples = filter_single_label_examples

        # if kwargs["tokenizer"] is None:
        #     self.vocab_file = vocab_file
        #     self.spm_model_file = spm_model_file
        #     tokenizer = FullTokenizer.from_scratch(vocab_file=vocab_file, do_lower_case=do_lower_case,
        #                                            spm_model_file=spm_model_file)

    # def prepare_model_data(self, data_type, batch_size=32, from_tfrecord=False):
    #     input_file = self.input_file_dict.get(data_type)
    #     tfrecord_input_file = f"{input_file}.{self.tfrecord_suffix}"
    #     if from_tfrecord and os.path.exists(tfrecord_input_file):
    #         # 读取 tf_record 数据，生成 data_generator                                                                                                                                                                                   output_file=input_file)
    #         dataset = self.get_dataset_from_tfrecord(data_type=data_type, batch_size=batch_size)
    #         return dataset
    #     elif os.path.exists(input_file):
    #         # 读取原始的examples
    #         examples_ls = load_json(input_file)
    #         examples = [JuyuanNerClassifierPipelineInputExampleV2(**example) for example in examples_ls]
    #         if self.debug:
    #             examples = examples[0:self.debug_example_num]
    #         logger.info(f"读取 {data_type} examples 共 {examples.__len__()} 个 from {input_file}")
    #         data_generator = self.create_generator(data_type, examples, batch_size)
    #         return data_generator
    #     else:
    #         all_examples = self.get_examples(data_type)
    #         os.makedirs(os.path.dirname(input_file), exist_ok=True)
    #         # 转换为 features
    #         features, examples, exception_examples = self._file_based_convert_examples_to_features(
    #             examples=all_examples, data_type=data_type, output_file=tfrecord_input_file)
    #         examples_ls = [example.to_dict() for example in examples]
    #         dump_json(examples_ls, json_path=input_file)
    #         # 转换为 dataset 或 generator
    #         data_generator = self.create_generator(data_type, examples, batch_size)
    #         return data_generator

    def prepare_model_data(self, data_type) -> KerasTextClassifierDataGenerator:
        # 读取 examples
        examples = self._get_examples(data_type=data_type)
        if examples:
            # 筛选最后输入模型的 examples
            filter_examples = self._filter_examples(examples=examples)
            logger.info(f"原有 {data_type} examples 共 {examples.__len__()} 个,过滤后的 examples 共 {filter_examples.__len__()}")
            # 对 输入模型的 examples 进行数据分析
            data_analysis = TextClassifierDataAnalysis(data_type=data_type, data_examples=filter_examples)

            # 动态设置 model_support_types_dict
            self._set_model_support_types_dict(data_type, label_counter=data_analysis.label_counter)

            # 加入 input_examples 的 特征
            input_examples = self._get_input_examples(examples=filter_examples)

            batch_size = self._get_batch_size(data_type=data_type)

            data_generator = self._create_generator(data_type, input_examples, batch_size,
                                                    data_analysis=data_analysis)

            return data_generator

    def _get_examples(self, data_type='train', if_convert_examples_to_features=False) -> List[
        JuyuanNerClassifierPipelineInputExampleV2]:
        """
        获取 JuyuanNerClassifierPipelineInputExampleV2
        """
        input_filepath = self.input_filepath_dict[data_type]
        raw_data_path = self.raw_filepath_dict[data_type]
        if os.path.exists(input_filepath):
            # 读取原始的examples
            examples_data_ls = load_json(input_filepath)
            examples = [self._create_juyuan_example(guid=example_data_dict.pop('guid'),
                                                    data_dict=example_data_dict)
                        for example_data_dict in examples_data_ls]
        elif if_convert_examples_to_features:
            # 从原始数据中读取所有的 examples
            all_examples = None
            # 转换为 features
            features, examples, exception_examples = self._file_based_convert_examples_to_features(
                examples=all_examples, data_type=data_type)
            examples_ls = [example.to_dict() for example in examples]
            dump_json(examples_ls, json_path=input_filepath)
        elif os.path.exists(raw_data_path):
            # TODO: 创建截取 entity周围文本后的 examples
            raw_data = dataframe_to_file(path=raw_data_path, mode="r")
            if self.debug:
                not_valid_example_num = 0
                examples = []
                for i in tqdm.tqdm(range(100), desc="创建 JuyuanNerClassifier example"):
                    row = raw_data.iloc[i]
                    example = self._create_juyuan_example(guid=row.name,
                                                          data_dict=row.to_dict())
                    if example.entity_context:
                        examples.append(example)
                    else:
                        not_valid_example_num += 1
            else:
                # 转换为  JuyuanNerClassifierPipelineInputExampleV2
                examples_series = raw_data.apply(lambda row: self._create_juyuan_example(guid=row.name,
                                                                                         data_dict=row.to_dict()),
                                                 axis=1)
                is_valid_example = examples_series.apply(
                    lambda example: True if example.entity_context is not None else False)
                examples = examples_series.loc[is_valid_example].tolist()

            examples_ls = [example.to_dict() for example in examples]
            dump_json(examples_ls, json_path=input_filepath)
        else:
            examples = None

        return examples

    def _create_juyuan_example(self, guid: int, data_dict: Dict[Text, Union[Text, List[Text]]]) \
            -> JuyuanNerClassifierPipelineInputExampleV2:
        forth_event_label = data_dict.pop(FORTH_EVENT_LABEL_COLUMN)
        third_event_label = data_dict.pop(THIRD_EVENT_LABEL_COLUMN)
        example = JuyuanNerClassifierPipelineInputExampleV2(guid=guid,
                                                            # support_types_dict=support_types_dict,
                                                            forth_event_label=forth_event_label,
                                                            third_event_label=third_event_label,
                                                            **data_dict)
        return example

    def _filter_examples(self, examples:JuyuanNerClassifierPipelineInputExampleV2)\
            -> List[JuyuanNerClassifierPipelineInputExampleV2]:
        valid_examples = []
        for example in examples:
            valid_example = self._filter_example(example)
            if valid_example:
                valid_examples.append(valid_example)
        return valid_examples

    def _filter_example(self, example: JuyuanNerClassifierPipelineInputExampleV2,
                                  ) -> JuyuanNerClassifierPipelineInputExampleV2:
        """
        转换模型的输入数据 InputExample
        """
        if self.use_third_event_label:
            labels = example.third_event_label
        else:
            labels = example.forth_event_label

        if self.train_data_highest_threshold:
            support_types = {support_type for support_type, value in self.total_support_types_dict.items()
                             if value[SUPPORT_TYPE_COUNT] < self.train_data_highest_threshold}
            labels = list(set(labels).intersection(support_types))

        if self.train_data_lowest_threshold:
            support_types = {support_type for support_type, value in self.total_support_types_dict.items()
                             if value[SUPPORT_TYPE_COUNT] > self.train_data_lowest_threshold}
            labels = list(set(labels).intersection(support_types))

        if self.filter_source:
            # 使用 source 对训练数据进行过滤
            if example.source not in self.filter_source:
                labels = None
        if self.filter_single_label_examples:
            if labels and labels.__len__()>1:
                labels  = None

        if labels:
            example.label = labels
            return example


    def _get_input_examples(self,examples:JuyuanNerClassifierPipelineInputExampleV2)\
            -> List[JuyuanNerClassifierPipelineInputExampleV2]:
        # 转换为 input_examples
        input_examples = []
        for example in examples:
            input_example = self._convert_to_input_example(example)
            input_examples.append(input_example)

        return input_examples


    def _convert_to_input_example(self, example: JuyuanNerClassifierPipelineInputExampleV2,
                                  ) -> JuyuanNerClassifierPipelineInputExampleV2:
        """
        转换模型的输入数据 InputExample
        """
        labels =  example.label
        label_ids = self._convert_multilabel2list(label_ls=labels, support_types_dict=self.model_support_types_dict)
        # 使用原来的example,在其中增加模型训练时候的属性
        assert np.sum(label_ids)>0
        example.text_a = example.entity_context
        example.label_id = label_ids
        return example


    @staticmethod
    def _convert_multilabel2list(label_ls: List[Text], support_types_dict: Dict[Text, int],
                                 label_index_key: Text = SUPPORT_TYPE_INDEX):
        """
        @author:wangfc27441
        @time:  2019/10/21  15:50
        @desc: 将 label_ls 转化为 一维数组: [0,0,...,1,0,...1,0,...0]
        """
        multilabel_indicator = np.zeros(shape=len(support_types_dict), dtype=np.int8)
        for label in label_ls:
            # 获取数据的标签，并将其转换为 multilabel_indicator，其他不在 support_types_dict中的标签返回为None
            label_index = support_types_dict.get(label, None)
            if isinstance(label_index, dict):
                label_index = label_index[label_index_key]
            if label_index is not None:
                multilabel_indicator[label_index] = 1
        return multilabel_indicator.tolist()

    def _create_generator(self, data_type: Text,
                          examples: JuyuanNerClassifierPipelineInputExampleV2,
                          batch_size: int,
                          data_analysis:TextClassifierDataAnalysis =None) \
            -> KerasTextClassifierDataGenerator:
        """
        创建 KerasTextClassifierDataGenerator
        JuyuanNerClassifierPipelineInputExampleV2 -> InputExample -> KerasTextClassifierDataGenerator
        """

        # if data_type == TRAIN_DATATYPE:
        #     self.train_data_analysis = TextClassifierDataAnalysis(data_type=data_type, data_examples=input_examples)

        # 创建 data_generator
        data_generator = KerasTextClassifierDataGenerator(data=examples, tokenizer=self.tokenizer,
                                                          data_type=data_type, max_seq_length=self.max_seq_length,
                                                          batch_size=batch_size, batch_strategy=self.batch_strategy,
                                                          examples=examples,
                                                          data_analysis = data_analysis
                                                          )

        return data_generator





    # def get_train_examples(self, data_dir, train_data_name='train.json'):
    #     """See base class."""
    #     # 原数据预处理：分割为训练集和测试集，获取多标签的数据
    #     train_path = os.path.join(data_dir, train_data_name)
    #     return self._create_multilabel_examples(train_path, 'train')

    def read_from_web_data(self, set_type, data):
        examples = []
        for index, sample in enumerate(data):
            guid = '%s-%s' % (set_type, index)
            ids = sample.get('id')
            title = sample.get('title', '')
            content = sample.get('content', '')
            entity = sample.get('entity')
            entity_word = sample.get('word')  # 增加聚源接口提供 word
            entity_nature = sample.get('nature')
            entity_start_position = sample.get('entity_start_position', -1)
            entity_end_position = sample.get('entity_end_position', -1)

            # 把字符串变成unicode的字符串。这是为了兼容Python2和Python3，
            # 因为Python3的str就是unicode，而Python2的str其实是bytearray，Python2却有一个专门的unicode类型。
            # title = tokenization.convert_to_unicode(title)
            # content = tokenization.convert_to_unicode(content)
            examples.append(
                JuyuanNerClassifierPipelineInputExampleV2(guid=guid, ids=ids,
                                                          title=title, content=content,
                                                          entity=entity,
                                                          entity_word=entity_word,
                                                          entity_nature=entity_nature,
                                                          entity_start_position=entity_start_position,
                                                          entity_end_position=entity_end_position,
                                                          event_label_index=np.zeros(len(self.support_types_dict))
                                                          # event_label_index =np.zeros(len(self.support_types_dict)) # 预测的时候实际没有作用
                                                          ))
            return examples

    def _get_striped_positions(self, title, content, entity_start_position, entity_end_position):
        raw_context = title + '\n' + content
        entity = raw_context[entity_start_position:entity_end_position + 1]
        # context_striped = title.strip() + '\n' + content.strip()
        # entity = context_striped[entity_start_position:entity_end_position + 1]
        striped_context = title.strip() + '\n' + content.strip()
        if entity_start_position < title.__len__():
            # entity 在 title 中
            title_left_strip_size = title.__len__() - title.lstrip().__len__()
            entity_start_position -= title_left_strip_size
            entity_end_position -= title_left_strip_size
        else:
            # entity 在 content 中
            title_strip_size = title.__len__() - title.strip().__len__()
            content_left_strip_size = content.__len__() - content.lstrip().__len__()
            strip_size = title_strip_size + content_left_strip_size
            entity_start_position -= strip_size
            entity_end_position -= strip_size
        entity_from_position = striped_context[entity_start_position:entity_end_position + 1]
        return title.strip(), content.strip(), entity_from_position, entity_start_position, entity_end_position

    def read_examples_from_json(self, set_type, data_path):
        # 从本地文件csv文件中读取数据
        df = pd.read_json(path_or_buf=data_path)
        logger.info(f"读取数据共{df.__len__()}条 from {data_path}")
        examples = []
        none_mathched_entity_num = 0
        if self.debug == True:
            examples_num = 100
        else:
            examples_num = len(df)
        for i in tqdm.tqdm(range(examples_num)):
            line = df.iloc[i]
            guid = '%s-%s' % (set_type, i)
            # example_index = line['index']
            id = line['id']  # '20200519006-000-000275'
            title = line.get('title').strip()
            content = line.get('content')
            source = line.get('source')
            if title is None:
                title = ""
            if content is None:
                content = ""
            else:
                content = content.strip()

            entity_label_info_ls = line.get('entity_label_info_ls')
            for entity_index, entity_label_info_dict in enumerate(entity_label_info_ls):
                entity = entity_label_info_dict.get('entity').strip()
                entity_start_position = int(entity_label_info_dict.get('start_position'))
                entity_end_position = int(entity_label_info_dict.get('end_position'))
                # title,content,entity_from_position, entity_start_position,entity_end_position = self.get_striped_positions(title, content, entity_start_position, entity_end_position)
                entity_from_position = (title + '\n' + content)[entity_start_position:entity_end_position + 1]

                event_label_ls = entity_label_info_dict.get('label')

                not_matched_entity = entity.lower() != entity_from_position.lower()
                if not_matched_entity:
                    if none_mathched_entity_num % 1000 == 0:
                        logger.warning(
                            '第{}个entity位置不匹配 entity={},entity_from_position={},source={}'.format(
                                none_mathched_entity_num, entity,
                                entity_from_position, source))
                    none_mathched_entity_num += 1
                    continue

                intersect_labels_set = set(event_label_ls).intersection(self.support_event_types_set)
                none_intersect_label = False
                multilabels = False
                if intersect_labels_set is None or intersect_labels_set == set():
                    none_intersect_label = True
                elif intersect_labels_set.__len__() > 1:
                    multilabels = True

                # 对于测试数据，我们过滤那些不在本次模型支持的范围内的数据 or 多标签 or entity 位置不对的数据
                if set_type == "test" and (none_intersect_label or multilabels or not_matched_entity):
                    continue
                else:
                    # if set_type == "test":
                    #     intersect_label_ls = list(intersect_labels_set)
                    # 因为聚源的生产数据中 entity 被匹配为全称或者证券名称，可能在原文中没有找到对应的实体位置，我们默认为 -1
                    examples.append(
                        # 这儿我们使用自定义的 NerClassifierPipelineInputExample 来表示原始数据
                        JuyuanNerClassifierPipelineInputExampleV2(guid=guid, id=id, title=title, content=content,
                                                                  entity_index=entity_index,
                                                                  entity=entity,
                                                                  entity_start_position=entity_start_position,
                                                                  entity_end_position=entity_end_position,
                                                                  event_label_ls=event_label_ls)
                    )
        logger.info('data_type={},examples={},none_mathched_entity_num={}'.format(set_type, examples.__len__(),
                                                                                  none_mathched_entity_num))
        return examples

    def _create_multilabel_examples(self, data_path=None, set_type='train', data=None, web_predict=False):
        """Creates examples for the training and dev sets."""
        if web_predict:
            # 从web端输入数据
            examples = self.read_from_web_data(set_type, data)
        else:
            # 从本地的json文件中读取
            examples = self.read_examples_from_json(set_type, data_path)
        return examples

    # def get_dev_examples(self, data_dir, dev_data_name='dev.json'):
    #     """See base class."""
    #     dev_path = os.path.join(data_dir, dev_data_name)
    #     return self._create_multilabel_examples(dev_path, 'dev')
    #
    # def get_test_examples(self, data_dir, test_data_name='test.json'):
    #     """See base class."""
    #     test_path = os.path.join(data_dir, test_data_name)
    #     return self._create_multilabel_examples(test_path, 'test')

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
        return self._create_multilabel_examples(data_path=None, set_type='web_predict', web_predict=True, data=data)

    def get_labels(self):
        """See base class."""
        support_event_types_dict_path = os.path.join(self.data_dir, 'support_types_dict.json')
        with open(support_event_types_dict_path, mode='r', encoding='utf-8') as f:
            self.support_types_dict = json.load(f)
            self.support_event_types_set = set(self.support_types_dict.keys())
        return list(self.support_types_dict.keys())

    def _file_based_convert_examples_to_features(
            self, examples: JuyuanNerClassifierPipelineInputExampleV2, data_type: Text, output_file=None):
        """Convert a set of `InputExample`s to a TFRecord file.
        Text:str -> tokens:List[str] -> token ids:List[int] -> feature:tf.train.Feature -> tf.train.Example -> tf.record
        """
        features = []
        examples_list = []
        # 存储异常的example
        exception_examples_list = []
        # writer = tf.python_io.TFRecordWriter(output_file)
        writer = tf.io.TFRecordWriter(output_file)

        for ex_index in tqdm.tqdm(range(examples.__len__())):
            example = examples[ex_index]
            if ex_index % 10000 == 0:
                tf.compat.v1.logging.info("Writing example %d of %d to %s" % (ex_index, len(examples), output_file))
            # 聚源数据集的example转换为feature
            feature, exception_example = self._convert_single_juyuan_ner_classifier_pipeline_exampleV3(ex_index,
                                                                                                       example)

            if exception_example is not None:
                exception_examples_list.append(exception_example)
                if len(exception_examples_list) % 10 == 0:
                    tf.compat.v1.logging.info('异常数据已经产生数量为：{}'.format(len(exception_examples_list)))
                continue

            else:
                # 增加 example 的属性
                example.input_ids = feature.input_ids
                example.input_mask = feature.input_mask
                example.segment_ids = feature.segment_ids
                example.label_ids = feature.label_ids

                examples_list.append(example)

                # 有序的字典
                # Create a dictionary mapping the feature name to the tf.Example-compatible data type
                features = collections.OrderedDict()
                features["input_ids"] = self._create_int_feature(feature.input_ids)
                features["input_mask"] = self._create_int_feature(feature.input_mask)
                features["segment_ids"] = self._create_int_feature(feature.segment_ids)
                features["label_ids"] = self._create_int_feature(feature.label_ids)
                features["is_real_example"] = self._create_int_feature([int(feature.is_real_example)])

                # Create a Features message using tf.train.Example.
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))

                # 使用 .SerializeToString 方法将所有协议消息序列化为二进制字符串 进行序列化并写入
                writer.write(tf_example.SerializeToString())
        writer.close()

        if exception_examples_list != []:
            exception_examples_path = os.path.join(self.data_dir, f'{data_type}.exception_examples')
            with open(exception_examples_path, mode='w', encoding='utf-8') as f:
                json.dump(exception_examples_list, f, ensure_ascii=False, indent=4)
                print('save exception_examples of {} into {}'.format(len(exception_examples_list),
                                                                     exception_examples_path))

        return features, examples_list, exception_examples_list

    def _convert_single_juyuan_ner_classifier_pipeline_exampleV3(self, ex_index, example,
                                                                 if_print=True,
                                                                 assure_entity_token_positon=False):
        """
        @author:wangfc27441
        @desc:
        @version：
        @time:2020/12/15 15:41


        :param ex_index:
        :param example:
        :param label_list:
        :param max_seq_length:
        :param tokenizer:
        :param if_print:
        :return:
        """
        feature = None
        exception_example = None
        try:
            title = example.title
            content = example.content
            entity = example.entity
            entity_start_position = example.entity_start_position
            entity_end_position = example.entity_end_position

            # 如果entity出现在 title中,对 tokens_b保留原样，并再后面的处理过程中从前往后裁剪
            if entity_start_position < len(title):
                content = content
                context = title + '\n' + content
            # 如果 entity出现在 content 中，对 token_b 进行裁剪：
            else:
                # 找到 该entity所在语句，从上一句的 句号以后位置开始截取
                last_sentence_position_in_content = self._find_last_sentence_end_segment_position(title, content,
                                                                                                  entity,
                                                                                                  entity_start_position,
                                                                                                  self.max_seq_length)
                context = content[last_sentence_position_in_content + 1:]
                # context = title + '\n' + content
                # 更新 entity_start_position
                entity_start_position = entity_start_position - (last_sentence_position_in_content + 1)
                entity_end_position = entity_end_position - (last_sentence_position_in_content + 1)
                # 验证新的 new_context
                entity_after_window = context[entity_start_position: entity_start_position + len(entity)].lower()
                if entity.lower() != entity_after_window:
                    raise (ValueError(
                        "截取窗口后的entity:{} != entity_after_window:{}".format(entity.lower(), entity_after_window)))

            # 对 content 进行阶段，防止过长的文档影响时间
            content = content[:5000]

            if assure_entity_token_positon:
                # 需要使用 entity 位置的情况
                chinese_basic_tokenizer = ChineseBaiscTokenizerV2()
                segments_a, char_to_segment_offset_a = chinese_basic_tokenizer.segment(title, absolute=True)
                segments_b, char_to_segment_offset_b = chinese_basic_tokenizer.segment(content, absolute=True)

                all_doc_tokens_a, tok_to_orig_index_a, orig_to_tok_index_a = get_subword_tokens(self.tokenizer,
                                                                                                segments_a)
                all_doc_tokens_b, tok_to_orig_index_b, orig_to_tok_index_b = get_subword_tokens(self.tokenizer,
                                                                                                segments_b)

                # start = entity_start_position-len(title)-1
                # end = entity_end_position -len(title)-1
                # entity_in_content = content[start:end+1]
                # seg_start_index = char_to_segment_offset_b[start]
                # segment_start = segments_b[seg_start_index]
                # seg_end_index = char_to_segment_offset_b[end]
                # segment_end = segments_b[seg_end_index]
                # segments_b[seg_start_index:seg_end_index+1]
                # token_start_index = orig_to_tok_index_b[seg_start_index]
                # token_end_index = orig_to_tok_index_b[seg_end_index]
                # all_doc_tokens_b[token_start_index:token_end_index+1]

                # 原始的 char position -> seg 的 position -> token 的 position
                entity_in_a = False
                entity_in_b = False
                if entity_start_position < title.__len__():
                    entity_token_start_position = orig_to_tok_index_a[char_to_segment_offset_a[entity_start_position]]
                    entity_token_end_position = orig_to_tok_index_a[char_to_segment_offset_a[entity_end_position]]
                    entity_in_a = True
                    self._assure_entity_token_position(entity, all_doc_tokens_a, entity_token_start_position,
                                                       entity_token_end_position)

                else:
                    entity_start_position_in_content = entity_start_position - len(title) - 1
                    entity_end_position_in_content = entity_end_position - len(title) - 1
                    entity_token_start_position = orig_to_tok_index_b[
                        char_to_segment_offset_b[entity_start_position_in_content]]
                    entity_token_end_position = orig_to_tok_index_b[
                        char_to_segment_offset_b[entity_end_position_in_content]]
                    entity_in_b = True
                    self._assure_entity_token_position(entity, all_doc_tokens_b, entity_token_start_position,
                                                       entity_token_end_position)

                tokens_a = all_doc_tokens_a
                tokens_b = all_doc_tokens_b
            else:
                tokens_a = self.tokenizer.tokenize(title)
                tokens_b = self.tokenizer.tokenize(content)

            # 对数据进行标准化的bert处理
            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > self.max_seq_length - 2:
                    tokens_a = tokens_a[0:(self.max_seq_length - 2)]

            tokens = []
            segment_ids = []
            entity_mask = []

            tokens.append("[CLS]")
            segment_ids.append(0)
            entity_mask.append(0)

            # 防止 title=''的情况
            index_a = 0
            for index_a, token in enumerate(tokens_a):
                tokens.append(token)
                segment_ids.append(0)
                # 获取 entity_mask
                if assure_entity_token_positon and entity_in_a and index_a > entity_token_start_position - 1 and index_a < entity_token_end_position + 1:
                    entity_mask.append(1)
                else:
                    entity_mask.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            entity_mask.append(0)

            if tokens_b:
                for index_b, token in enumerate(tokens_b):
                    tokens.append(token)
                    segment_ids.append(1)
                    index = index_a + 1 + index_b
                    if assure_entity_token_positon and entity_in_b and index_b > entity_token_start_position - 1 and index < entity_token_end_position + 1:
                        entity_mask.append(1)
                    else:
                        entity_mask.append(0)

                tokens.append("[SEP]")
                segment_ids.append(1)
                entity_mask.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                entity_mask.append(0)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(entity_mask) == self.max_seq_length

            event_label_ls = example.event_label_ls
            label_id = self._convert_multilabel2list(label_ls=event_label_ls,
                                                     support_types_dict=self.support_types_dict)

            if ex_index < 5 and if_print:
                tf.compat.v1.logging.info("*** Example ***")
                tf.compat.v1.logging.info("guid: %s" % (example.guid))
                min_context = min(len(context), 100)
                # tf.compat.v1.logging.info('title: %s' % (title))
                tf.compat.v1.logging.info('context[:100]: %s' % (context[:min_context]))
                tf.compat.v1.logging.info('entity: %s' % (entity))
                tf.compat.v1.logging.info('event_label_ls: %s' % (example.event_label_ls))
                tf.compat.v1.logging.info("tokens: %s" % " ".join(tokens))
                tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                # tf.compat.v1.logging.info("label: %s (id = %d)" % (example.label, label_id))
                tf.compat.v1.logging.info("label: %s " % " ".join([str(x) for x in label_id]))
                tf.compat.v1.logging.info('entity_mask: %s' % " ".join([str(x) for x in entity_mask]))

            feature = InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_id,
                is_real_example=True,
                entity_mask=entity_mask
            )
        except Exception as e:
            logger.error(e)
            exception_example = example.__dict__
        finally:
            return feature, exception_example

    @staticmethod
    def _find_last_sentence_end_segment_position(title, content, entity_start_position,
                                                 title_content_joined_char=TITLE_CONTENT_JOINED_CHAR,
                                                 sentence_end_segment=SENTENCE_END_SEGMENT):
        """
        @author:wangfc27441
        @desc:  简化 position 的位置查找
        @version：
        @time:2020/12/14 11:20

        Parameters
        ----------

        Returns
        -------
        """
        # 查找的分割符号
        # entity_start_position_in_content： entity所在content的的开始位置
        # 因为增加了在 title 和content 中间增加了 一个 '\n'
        entity_start_position_in_content = entity_start_position - len(title) - len(title_content_joined_char)
        # assert entity == content[entity_start_position_in_content: entity_start_position_in_content + len(entity)]

        # 从 entity_start_position_in_content 开始反向查找到首个 seg的位置
        last_sentence_position_in_content = content.rfind(sentence_end_segment, 0, entity_start_position_in_content)

        """
        # 截取后的包含entity的最小的 content_window的最小长度： 前面的正文 + entity + 128 （默认）
        # 窗口开始的位置
        window_start_position = last_sentence_position_in_content
        # 窗口结束的位置
        window_end_position =  entity_start_position_in_content + len(entity) + after_entity_content_window_length
        # least_entity_content_window_length = entity_start_position_in_content - last_sentence_position_in_content + len(
        #     entity) + entity_content_window_length
        # 为简化计算，我们使用 window_length 来代替  window_tokens_length
        window_tokens_length = window_end_position - window_start_position
        # 如果超过最大长度
        if len(title) + window_tokens_length > max_seq_len and entity_start_position_in_content > after_entity_content_window_length-1:
            # 向前截取 after_entity_content_window_length 个位置作为截取的起点
            last_sentence_position_in_content = entity_start_position_in_content - after_entity_content_window_length
        """
        return last_sentence_position_in_content

    # 验证 entity token 位置是否正确
    def _assure_entity_token_position(self, entity, all_doc_tokens, entity_token_start_position,
                                      entity_token_end_position):
        entity_tokens_from_position = [entity_token for entity_token in
                                       all_doc_tokens[entity_token_start_position:entity_token_end_position + 1]]
        entity_tokens_from_position_jioned = ''.join(
            [entity_token.replace('##', '') for entity_token in entity_tokens_from_position])
        # 原始的去除英文表达中的空格并小写
        entity_joined = ''.join(entity.lower().split())
        try:
            assert entity_joined == entity_tokens_from_position_jioned
        except Exception as e:
            raise ValueError('原始的 entity不同于 entity_tokens_from_position： {} != {}'.format(entity, " ".join(
                entity_tokens_from_position)))





def columns_mapping(columns: List[Text],column_mapping = {"label": FORTH_EVENT_LABEL_COLUMN}):
    new_columns = []
    for column in columns:
        if column in column_mapping:
            new_column = column_mapping[column]
        else:
            new_column = column
        new_columns.append(new_column)
    return new_columns