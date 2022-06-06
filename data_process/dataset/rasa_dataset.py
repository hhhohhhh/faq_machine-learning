#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/26 9:18 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/26 9:18   wangfc      1.0         None
"""
import os
from collections import defaultdict
from typing import List, Any, Text, Union, Dict, Tuple
import tqdm

from data_process.data_utils import is_rasa_retrieve_intent, Action
from data_process.dataset.hsnlp_faq_knowledge_dataset import EntityData, IntentEntityInfoToResponse, \
    StandardQuestionKnowledgeData, ATTRIBUTE
from data_process.data_labeling import entity_labeling_with_tire_tree, EntityLabeler

import pandas as pd

from utils.constants import INTENT, SUB_INTENT, RESPONSE, QUESTION
from utils.io import write_to_yaml, read_yaml_file, _to_yaml_examples_dict, _to_yaml_string_dict, read_json_file, \
    dump_obj_as_json_to_file

import logging

logger = logging.getLogger(__name__)

RASA_VERSION = "2.0"
RASA_NLU_DATATYPE = 'nlu'
RASA_RESPONSES_DATATYPE = "responses"
RASA_RULES_DATATYPE = "rules"
RASA_STORIES_DATATYPE = "stories"

RASA_VOCABULARY_DATATYPE = 'vocabulary'
RASA_ENTITY_SYNONYM_DATATYPE = 'entity_synonym'
RASA_ENTITY_REGEX_DATATYPE = 'entity_regex'
RASA_INTENT_REGEX_DATATYPE = 'intent_regex'

RASA_DOMAIN_DATATYPE = 'domain'
RASA_NLU_DATA_FILENAME = f"{RASA_NLU_DATATYPE}.yml"
RASA_RULES_DATA_FILENAME = f"{RASA_RULES_DATATYPE}.yml"
RASA_STORIES_DATA_FILENAME = f"{RASA_STORIES_DATATYPE}.yml"
RASA_RESPONSES_DATA_FILENAME = f"{RASA_RESPONSES_DATATYPE}.yml"
RASA_DOMAIN_DATA_FILENAME = f"{RASA_DOMAIN_DATATYPE}.yml"
RASA_VOCABULARY_FILENAME = f"{RASA_VOCABULARY_DATATYPE}.yml"
RASA_ENTITY_SYNONYM_FILENAME = f"{RASA_ENTITY_SYNONYM_DATATYPE}.yml"
RASA_ENTITY_REGEX_FILENAME = f"{RASA_ENTITY_REGEX_DATATYPE}.yml"
RASA_INTENT_REGEX_FILENAME = f"{RASA_INTENT_REGEX_DATATYPE}.yml"

RASA_CONFIG_FILENAME = 'config.yml'


class RasaDataset():
    """
    将原始数据转换为 rasa 的 数据： nlu.yml, stories.yml,response.yml ,
    nlu.yml: 用于 NLU (意图识别 + 实体抽取)
    stories.yml :  用于 train your assistant's dialogue management model 定义 action(response)
    response.yml : 用于 bot 输出的语句
    """

    def __init__(self, data, output_dir, subdata_dir="data",
                 rasa_config_filename=RASA_CONFIG_FILENAME,
                 nlu_data_filename=RASA_NLU_DATA_FILENAME,
                 responses_data_filename=RASA_RESPONSES_DATA_FILENAME,
                 rules_data_filename=RASA_RULES_DATA_FILENAME,
                 stories_data_filename=RASA_STORIES_DATA_FILENAME,
                 domain_data_filename=RASA_DOMAIN_DATA_FILENAME,
                 vocabulary_filename=RASA_VOCABULARY_FILENAME,
                 entity_synonym_filename=RASA_ENTITY_SYNONYM_FILENAME,
                 rasa_entity_regex_filename=RASA_ENTITY_REGEX_FILENAME,
                 rasa_intent_regex_filename=RASA_INTENT_REGEX_FILENAME

                 ):
        self.data = data
        self.output_dir = output_dir
        self.rasa_config_filename = rasa_config_filename
        self.subdata_dir = subdata_dir
        self.output_data_dir = os.path.join(self.output_dir, self.subdata_dir)

        self.nlu_data_filename = nlu_data_filename
        self.responses_data_filename = responses_data_filename
        self.rules_data_filename = rules_data_filename
        self.stories_data_filename = stories_data_filename
        self.domain_data_filename = domain_data_filename
        self.vocabulary_filename = vocabulary_filename
        self.entity_synonym_filename = entity_synonym_filename
        self.entity_regex_filename = rasa_entity_regex_filename
        self.intent_regex_filename = rasa_intent_regex_filename

        self.config_path = os.path.join(self.output_dir, self.rasa_config_filename)

        self.nlu_data_path = os.path.join(self.output_data_dir, self.nlu_data_filename)
        self.rules_data_path = os.path.join(self.output_data_dir, self.rules_data_filename)
        self.stories_data_path = os.path.join(self.output_data_dir, self.stories_data_filename)
        self.responses_data_path = os.path.join(self.output_data_dir, self.responses_data_filename)
        self.domain_data_path = os.path.join(self.output_dir, self.domain_data_filename)

        self.vocabulary_path = os.path.join(self.output_data_dir, self.vocabulary_filename)
        # entity synonym data
        self.entity_synonym_data_path = os.path.join(self.output_data_dir, self.entity_synonym_filename)

        # entity regex data
        self.entity_regex_data_path = os.path.join(self.output_data_dir, self.entity_regex_filename)

        # intent regex data
        self.intent_regex_data_path = os.path.join(self.output_data_dir, self.intent_regex_filename)

    def transform_to_nlu_data(self):
        raise NotImplementedError

    def transform_to_response_data(self):
        pass

    def transform_to_stories_data(self):
        """
        使用 output from the NLU pipeline （intent and entities）进行 action

        """
        pass

    def get_domain_data(self):
        """
        主要包括 version，intents, entities, slots, actions,forms
        """
        pass

    def to_rasa_format(self, data_ls: Union[List[Any], Dict[Text, Any]],
                       datatype: Text,
                       version=RASA_VERSION):
        # 只选取 前topk 做实验
        data = {"version": version, datatype: data_ls}
        return data

    def get_model_config(self):
        if os.path.exists(self.config_path):
            model_config = read_yaml_file(self.config_path)
            return model_config


class IntentDataToRasaDataset(RasaDataset):
    def __init__(self, raw_data: pd.DataFrame,
                 vocabulary: List[Text] = None,
                 output_dir=None,
                 rasa_config_filename=None,
                 data=None,
                 subdata_dir='data',
                 response_key=SUB_INTENT,
                 intent_column=INTENT,
                 sub_intent_column=SUB_INTENT,
                 question_column=QUESTION,
                 use_retrieve_intent=False,
                 use_intent_column_as_label=True,
                 use_label_as_response=False,
                 entity_attribute_to_value_dict: Dict[Text, List[Text]] = None,

                 if_get_core_data=False, if_get_stories_data=False,
                 if_get_vocabulary_data=False, if_get_synonym_data=False,
                 **kwargs
                 ):
        super(IntentDataToRasaDataset, self).__init__(data=data, output_dir=output_dir,
                                                      rasa_config_filename=rasa_config_filename,
                                                      subdata_dir=subdata_dir)

        # 输入 原始的意图数据，需要变换 意图标签 和 实体识别标注
        self.data = raw_data
        self.intent_column = intent_column
        self.sub_intent_column = sub_intent_column
        self.question_column = question_column

        self.response_key = response_key

        # ivr 数据转换为 rasa 数据的时候不使用  retrieve_intent，并将 sub_intent_column 作为 label
        self.use_retrieve_intent = use_retrieve_intent
        self.use_intent_column_as_label = use_intent_column_as_label
        self.use_label_as_response = use_label_as_response

        # vocabulary 数据为 CounterVectorFeaturizer 建立词典的时候使用，可以融入业务中的一个重要词汇作为特征
        self.vocabulary_ls = vocabulary

        # 用于标注 entity
        self.entity_attribute_to_value_dict = entity_attribute_to_value_dict

        self.entity_data = EntityData()

        self.if_get_core_data = if_get_core_data
        self.if_get_stories_data = if_get_stories_data
        self.if_get_vocabulary_data = if_get_vocabulary_data
        self.if_get_synonym_data = if_get_synonym_data

        # entities 和 slots
        # self.entities, self.slots = self._get_entities_slots()

        # self.transform_to_rasa_data()

    def _get_entities_slots(self):
        """
        创建 entity 对应的 slot 信息
        """
        raise NotImplementedError

    # def _apply_entity_labeling(self, data):
    #     # 合并意图 + 增加 entity 信息
    #     # 1. 合并 科创板开通 和 创业板开通 到 开户查询，并且标注 科创板和创业板为的实体属性为 block
    #     # 2. 合并 查个股行情 和 查指数行情 到 查行情，并且标注 stock 和 zsstock
    #     # 3. 变换 手机软件下载 和 电脑软件下载 为检索式标签 软件下载/手机软件下载 和 软件下载/电脑软件下载，使用检索模型进行分类
    #     new_texts = data.loc[:, self.question_column].apply(
    #         lambda text: entity_labeling(text, self.entity_attribute_to_value_dict))
    #     data.loc[:, self.question_column] = new_texts
    #     return data

    def transform_to_rasa_data(self, if_get_rule_data=False,
                               if_get_stories_data=False,
                               if_get_vocabulary_data=False,
                               if_get_synonym_regex_data=False):
        if if_get_synonym_regex_data:
            self.entity_synonym_data, self.entity_regex_data, self.intent_rege_data = \
                self.transform_to_synonym_regex_data()

        self.nlu_data, self.intents ,self.intent_to_attribute_mapping = self.transform_to_nlu_data()

        # 读取配置文件
        if if_get_rule_data or if_get_stories_data:
            self.model_config = self.get_model_config()

        if if_get_rule_data:
            self.rules_data, actions = self.transform_to_rules_data()
        if if_get_stories_data:
            # 多轮对话的时候，使用 stories 来组织
            self.stories_data = self.transform_to_stories_data()
        if if_get_rule_data or if_get_stories_data:
            # 首先构建 rules 或者 story 获取所有的 actions，在获取 responses_data
            self.actions, self.responses_data = self.transform_to_response_data(actions)
            self.domain_data = self.transform_to_domain_data()

        if if_get_vocabulary_data:
            self.vocabulary = self.transform_to_vocabulary_data()

    def transform_to_vocabulary_data(self, vocabulary: List[Text] = None):
        if os.path.exists(self.vocabulary_path):
            rasa_vocabulary = read_yaml_file(filename=self.vocabulary_path)
        else:
            assert isinstance(self.vocabulary_ls, list)
            rasa_vocabulary = self.to_rasa_format(self.vocabulary_ls, datatype=RASA_VOCABULARY_DATATYPE)
            write_to_yaml(path=self.vocabulary_path, data=rasa_vocabulary)
        return rasa_vocabulary

    def transform_to_synonym_regex_data(self):
        """
        获取 实体的近似词和 正则表达式， 意图的正则表达式
        """
        if os.path.exists(self.entity_synonym_data_path):
            rasa_entity_synonym_data = read_yaml_file(self.entity_synonym_data_path)
            rasa_entity_regex_data = read_yaml_file(self.entity_regex_data_path)
            rasa_intent_regex_data = read_yaml_file(self.intent_regex_data_path)
        else:
            rasa_entity_synonym_data, rasa_entity_regex_data, rasa_intent_regex_data = None, None, None
            entity_synonym_data, entity_regex_data, intent_regex_data = self._build_synonym_regex_data()
            # synonym_data 应该在 nlu 数据中
            if entity_synonym_data:
                rasa_entity_synonym_data = self.to_rasa_format(entity_synonym_data, datatype=RASA_NLU_DATATYPE)
                write_to_yaml(path=self.entity_synonym_data_path, data=rasa_entity_synonym_data)
            if entity_regex_data:
                rasa_entity_regex_data = self.to_rasa_format(entity_regex_data, datatype=RASA_NLU_DATATYPE)
                write_to_yaml(path=self.entity_regex_data_path, data=rasa_entity_regex_data)
            if intent_regex_data:
                rasa_intent_regex_data = self.to_rasa_format(intent_regex_data, datatype=RASA_NLU_DATATYPE)
                write_to_yaml(path=self.intent_regex_data_path, data=rasa_intent_regex_data)

        return rasa_entity_synonym_data, rasa_entity_regex_data, rasa_intent_regex_data

    def _build_synonym_regex_data(self) -> Tuple[
        List[Dict[Text, Text]], List[Dict[Text, Text]], List[Dict[Text, Text]]]:
        """
        构建 近义词，实体正则表达式，意图正则表达式
        """
        entity_synonym_data = self.entity_data._build_entity_synonym_data()
        entity_regex_data = self.entity_data._build_entity_regex_data()
        intent_regex_data = None
        return entity_synonym_data, entity_regex_data, intent_regex_data

    def transform_to_nlu_data(self, if_entity_labeling=True):
        intent_to_attribute_mapping = None
        intent_to_attribute_mapping_path = os.path.join(os.path.dirname(self.output_data_dir),
                                                        'intent_to_attribute_mapping.json')
        if os.path.exists(self.nlu_data_path):
            rasa_nlu_data = read_yaml_file(self.nlu_data_path)
            intents = [intent_to_examples.get('intent') for intent_to_examples in rasa_nlu_data.get(RASA_NLU_DATATYPE)]

            if os.path.exists(intent_to_attribute_mapping_path):
                intent_to_attribute_mapping = read_json_file(json_path=intent_to_attribute_mapping_path)
        else:
            data = self.data.copy()
            if if_entity_labeling:
                # TODO : 增加 entity 的标注： 首先验证 entity 是否存在于
                # 对数据进行实体标注:对 question 字段进行标注，并且获取 entity的信息
                entity_labeler = EntityLabeler(entity_data=self.entity_data)
                data = entity_labeler.apply_entity_labeling(data=data,
                                                            text_column=QUESTION)

            # 转换数据
            rasa_nlu_data_ls, intents ,intent_to_attribute_mapping = self._transform_raw_intent_data_to_rasa_nlu_data(
                raw_intent_data=data,
                intent_column=self.intent_column,
                sub_intent_column=self.sub_intent_column,
                question_column=self.question_column,
                use_retrieve_intent=self.use_retrieve_intent,
                use_intent_column_as_label=self.use_intent_column_as_label)

            rasa_nlu_data = self.to_rasa_format(rasa_nlu_data_ls, datatype=RASA_NLU_DATATYPE)
            write_to_yaml(path=self.nlu_data_path, data=rasa_nlu_data)

            if intent_to_attribute_mapping:
                dump_obj_as_json_to_file(filename=intent_to_attribute_mapping_path,obj= intent_to_attribute_mapping)

        return rasa_nlu_data, intents,intent_to_attribute_mapping

    def transform_to_response_data(self, actions=None) -> [List[Action], Union[List[Any], Dict[Text, Any]]]:
        if os.path.exists(self.responses_data_path):
            rasa_responses_data = read_yaml_file(self.responses_data_path)
            rasa_responses_data_dict = rasa_responses_data.get(RASA_RESPONSES_DATATYPE)
        else:
            if actions is not None:
                rasa_responses_data_dict = self._transform_rasa_response_data_from_actions(actions=actions)
            else:
                rasa_responses_data_dict = self._transform_raw_intent_data_to_rasa_response_data(
                    raw_intent_data=self.data,
                    actions=actions,
                    intent_column=self.intent_column,
                    sub_intent_column=self.sub_intent_column,
                    response_key=self.response_key,
                    use_retrieve_intent=self.use_retrieve_intent,
                    use_intent_column_as_label=self.use_intent_column_as_label,
                    use_label_as_response=self.use_label_as_response
                )

            rasa_responses_data = self.to_rasa_format(rasa_responses_data_dict, datatype=RASA_RESPONSES_DATATYPE)
            write_to_yaml(path=self.responses_data_path, data=rasa_responses_data)

        # 每个 response key 可以作为 action
        # actions = list(rasa_responses_data_dict.keys())
        actions = self._transform_responses_data_dict_to_actions(rasa_responses_data_dict)

        return actions, rasa_responses_data

    def transform_to_rules_data(self):
        from rasa.shared.utils.io import write_yaml

        if os.path.exists(self.rules_data_path):
            rasa_rules_data = read_yaml_file(self.rules_data_path)
            actions = None
        else:
            rasa_rules_ls, actions = self._transform_raw_intent_data_to_rules_data()
            rasa_rules_data = self.to_rasa_format(data_ls=rasa_rules_ls, datatype=RASA_RULES_DATATYPE)
            # write_to_yaml(path=self.rules_data_path, data=rasa_rules_data)
            write_yaml(data=rasa_rules_data, target=self.rules_data_path)
        return rasa_rules_data, actions

    def transform_to_domain_data(self):
        if os.path.exists(self.domain_data_path):
            rasa_domain_data = read_yaml_file(self.domain_data_path)
        else:
            rasa_domain_data = self._get_domain_data()
            write_to_yaml(path=self.domain_data_path, data=rasa_domain_data)
        return rasa_domain_data

    def _get_domain_data(self):
        domain_data = dict()
        action_names = [action.action_name for action in self.actions]
        domain_data.update({"version": RASA_VERSION,
                            "intents": self.intents,
                            "entities": self.entity_data._entity_types,
                            "slots": self.entity_data._slots,
                            "actions": action_names})
        return domain_data

    def _transform_raw_intent_data_to_rasa_nlu_data(self, raw_intent_data,
                                                    intent_column=INTENT,
                                                    sub_intent_column=SUB_INTENT,
                                                    question_column=QUESTION,
                                                    use_retrieve_intent=False,
                                                    use_intent_column_as_label=True):
        """
        use_retrieve_intent:  使用  intent = f"{intent}/{sub_intent}" 来构建 rasa retrieve_intent
        use_intent_column_as_label: Ture 的时候使用 intent 作为 label ,False 的时候使用 sub_intent 作为 label
        """
        from ruamel.yaml.scalarstring import PreservedScalarString as PSS

        rasa_nlu_data_ls = []
        intents = []
        nlu_example_num = 0
        if use_retrieve_intent:
            sub_intent_data = raw_intent_data.loc[raw_intent_data.loc[:, sub_intent_column].notna()].copy()
            none_sub_intent_data = raw_intent_data.loc[raw_intent_data.loc[:, sub_intent_column].isna()].copy()
            assert sub_intent_data.shape[0] + none_sub_intent_data.shape[0] == raw_intent_data.shape[0]
            # 对于具有子意图的数据
            grouped_sub_intent_data = sub_intent_data.groupby(by=[intent_column, sub_intent_column])
            for (intent, sub_intent), group in grouped_sub_intent_data:
                examples = group.loc[:, question_column].tolist()
                intent = f"{intent}/{sub_intent}"
                examples_str = "".join([f"- {example}\n" for example in examples])
                # 使用  PreservedScalarString: https://mlog.club/article/1462558
                nlu_data = {"intent": intent, "examples": PSS(examples_str)}
                rasa_nlu_data_ls.append(nlu_data)
                nlu_example_num += examples.__len__()
                intents.append(intent)

            # 对于没有子意图的数据
            grouped_none_sub_intent_data = none_sub_intent_data.groupby(by=[intent_column])
            for intent, group in grouped_none_sub_intent_data:
                intent_classifier_data = _to_yaml_examples_dict(key='intent', value=intent, examples=examples)
                rasa_nlu_data_ls.append(intent_classifier_data)
                nlu_example_num += examples.__len__()
                intents.append(intent)
        else:
            if use_intent_column_as_label:
                label_column = intent_column
            else:
                label_column = sub_intent_column

            # 对于所有的数据按照意图分组
            grouped_none_sub_intent_data = raw_intent_data.groupby(by=label_column)
            for intent, group in grouped_none_sub_intent_data:
                examples = group.loc[:, question_column].tolist()
                # 对 examples 进行排序： 文本长度+首字符
                # examples = sorted(examples, key=lambda x: (len(x), x.lower()))
                # intent = f"{intent}"
                # examples_str = "".join([f"- {example}\n" for example in examples if example.strip() != ''])
                # # 使用  PreservedScalarString: https://mlog.club/article/1462558
                # intent_classifier_data = {"intent": intent, "examples": PSS(examples_str)}
                intent_classifier_data = _to_yaml_examples_dict(key='intent', value=intent, examples=examples)

                rasa_nlu_data_ls.append(intent_classifier_data)
                intents.append(intent)
                nlu_example_num += examples.__len__()

        assert nlu_example_num == raw_intent_data.shape[0]
        return rasa_nlu_data_ls, intents

    def _apply_entity_labeling(self, data):
        """
        # 增加 entity 的标注：

        """
        # entity_attribute_to_value_dict = self.entity_attribute_to_value_dict.copy()
        # 首先 model_config 中是否存在 RegexEntityExtractor component，
        # 如果存在需要验证 是否存在 entity 使用 正则表达式 来 确定 entity 中，如果存在应该在 nlu标注的时候忽略
        # entity_attributes = self._get_entity_attributes_in_regex_entity_extractor()
        # for entity_attribute in entity_attributes:
        #     if entity_attribute in entity_attribute_to_value_dict:
        #         entity_attribute_to_value_dict.pop(entity_attribute)

        # 根据 清理的  entity_attribute_to_value_dict 建立 entity_tire_tree
        # self.entity_tire_tree = self._build_entity_tire_tree(entity_attribute_to_value_dict)

        new_texts = []
        for index in tqdm.tqdm(range(data.__len__()), desc='实体标注'):
            text = data.iloc[index].loc[self.question_column]
            new_text = entity_labeling_with_tire_tree(text, self.entity_data._entity_tire_tree)
            new_texts.append(new_text)
        # new_texts = data.loc[:, self.question_column].apply(
        #     lambda text: entity_labeling_with_tire_tree(text,self.entity_tire_tree))
        data.loc[:, self.question_column] = new_texts
        return data

    def _get_entity_attributes_in_regex_entity_extractor(self) -> List[Text]:
        entity_attributes = []
        if self._is_component_in_pipeline(component_name='RegexEntityExtractor'):
            entity_attributes = self._get_entity_attributes_in_regex_expression()
        return entity_attributes

    def _is_component_in_pipeline(self, component_name):
        """
        判断 component_name 是否存在 config 文件的 pipeline 中
        """
        if self.model_config:
            pipeline = self.model_config.get('pipeline')
            for component_config in pipeline:
                name = component_config.get('name')
                if name.lower() == component_name.lower():
                    return True
        return False

    def _get_entity_attributes_in_regex_expression(self) -> List[Text]:
        """
        获取正则表达式中所有的值
        regex : value
        """
        # 查询所有 regex 表达式作为可以对应的 value
        regex_key_values = self._get_key_values(key_name='regex')
        # 找出其中的 entity属性的 value
        entity_attributes_in_regex_expression = set(regex_key_values).intersection(
            set(self.entity_attribute_to_value_dict.keys()))

        return list(entity_attributes_in_regex_expression)

    def _get_key_values(self, key_name='regex'):
        """
        获取 key_name 对应的 值
        """
        key_values = []
        if self.synonym_data:
            for data in self.synonym_data.get('nlu'):
                keys = set(data.keys()).difference({'examples'})
                assert keys.__len__() == 1
                key = list(keys)[0]
                key_value = data.get(key)
                if key == key_name:
                    key_values.append(key_value)
        return key_values

    def _get_response_data_ls(self, raw_intent_data,
                              intent_column=INTENT,
                              sub_intent_column=SUB_INTENT,
                              response_key=RESPONSE,
                              use_retrieve_intent=False,
                              use_intent_column_as_label=True,
                              use_label_as_response=True) -> List[Dict[Text, Text]]:
        """
        获取 每个意图的 response_data: [{"intent": intent, "response": response},...]

        """
        response_data_ls = []
        if use_retrieve_intent:
            sub_intent_data = raw_intent_data.loc[raw_intent_data.loc[:, sub_intent_column].notna()].copy()
            none_sub_intent_data = raw_intent_data.loc[raw_intent_data.loc[:, sub_intent_column].isna()].copy()
            assert sub_intent_data.shape[0] + none_sub_intent_data.shape[0] == raw_intent_data.shape[0]
            # 对于具有子意图的数据
            grouped_sub_intent_data = sub_intent_data.groupby(by=[intent_column, sub_intent_column])
            for (intent, sub_intent), group in grouped_sub_intent_data:
                # 为每个意图设置 response, 使用 response_key 对应的字段作为 response
                if response_key == INTENT:
                    response = intent
                elif response_key == SUB_INTENT:
                    response == sub_intent
                else:
                    response = group.iloc[0].loc[response_key]
                response_data_ls.append({"intent": intent, "response": response})
        else:
            if use_intent_column_as_label:
                label_column = intent_column
            else:
                label_column = sub_intent_column

            # 对于所有的数据按照意图分组
            grouped_none_sub_intent_data = raw_intent_data.groupby(by=label_column)
            for intent, group in grouped_none_sub_intent_data:
                # 为每个意图设置 response, 使用 response_key 对应的字段作为 response
                # if response_key == intent_column:
                #     response = intent
                #     # raise (f"当 use_intent_column_as_label 为 False 的时候，response_key不能设置为 {INTENT}")
                # elif response_key == sub_intent_column:

                # 使用 label 作为 response
                if use_label_as_response:
                    response = intent
                else:
                    # 使用数据中的 response
                    response = group.iloc[0].loc[response_key]
                response_data_ls.append({"intent": intent, "response": response})
        return response_data_ls

    def _update_response_with_intent_entity_info_mapping(self, intent, rasa_responses_data_dict):

        """
        我们使用 intent_entity_info_mapping 来生成 response, self.data 是已经变换 意图后的数据,
        构建每个 意图 + slot 对应的 response action ( utter_*** or custom action)

        转换后的意图有 1） 合并的 2） 检索型的，
        1. 对于 简单的意图:
            action = utter_intent
            response_text = ...

        2. 对于 合并的意图 ，需要增加 action: 如果需要根据 entity 区分不同的 action
            1）对于不需要使用 slot 进行判断的 action，同 1） 简单意图
            2）需要使用 slot 进行判断的 action：
            action = utter_intent+slot (或者对应的 sub_intent)
            response_text =  自定义 (或者对应的 sub_intent)


        3. 对于检索型的意图，直接返回 response:
            action =  utter_retrieve_intent
            response_text= retrieve_intent_key (或者对应的 sub_intent)

        """
        from rasa.nlu.constants import RESPONSE_IDENTIFIER_DELIMITER

        if intent in self.intent_entity_info_mapping and intent.split(RESPONSE_IDENTIFIER_DELIMITER).__len__() == 1:
            entity_info_mappings = self.intent_entity_info_mapping.get(intent)
            for entity_info_mapping in entity_info_mappings:
                raw_label = entity_info_mapping.get('raw_label')
                action_name = f"utter_{raw_label}"
                response_text = raw_label
                rasa_responses_data_dict.update({action_name: [{'text': response_text}]})
        elif intent in self.intent_entity_info_mapping and intent.split(
                RESPONSE_IDENTIFIER_DELIMITER).__len__() == 2:
            action_name = f"utter_{intent}"
            response_text = intent.split(RESPONSE_IDENTIFIER_DELIMITER)[1]
            rasa_responses_data_dict.update({action_name: [{'text': response_text}]})
        return rasa_responses_data_dict

    def _transform_raw_intent_data_to_rasa_response_data(self, raw_intent_data,
                                                         actions=None,
                                                         intent_column=INTENT,
                                                         sub_intent_column=SUB_INTENT,
                                                         response_key=RESPONSE,
                                                         use_retrieve_intent=True,
                                                         use_intent_column_as_label=True,
                                                         use_label_as_response=False
                                                         ):
        raise NotImplementedError

    def _transform_rasa_response_data_from_actions(self, actions: List[Action] = None):
        """
        从 self.actions 中获取所有的 rasa_responses_data_dict
        """
        rasa_responses_data_dict = {}
        assert actions is not None and actions != []
        for action in actions:
            rasa_responses_data_dict.update({action.action_name: [{'text': action.response_text}]})
        return rasa_responses_data_dict

    def _transform_responses_data_dict_to_actions(self, responses_data_dict: Dict[Text, List[Dict[Text, Text]]]) -> \
            List[Action]:
        actions = []
        for action_name, response_text_ls in responses_data_dict.items():
            action = Action(action_name, response_text=response_text_ls[0])
            actions.append(action)
        return actions

    # def _is_retrieve_intent(self, intent):
    #     return intent.split(RESPONSE_IDENTIFIER_DELIMITER).__len__() == 2

    # def _make_action(self, intent, use_custom_actions=True):
    #     is_intent_a_retrieve_intent = self._is_retrieve_intent(intent)
    #     if is_intent_a_retrieve_intent:
    #         """
    #         rasa retrieve_intent: 我们构建 response = f"utter_{intent}/{sub_intent}"
    #         参考 https://rasa.com/docs/rasa/chitchat-faqs
    #         对于 retrieval_intent ,我们只需要对 retrival intent 生成 rule
    #         因为  self.intents 包括 retrieval_intents + retrieval_intent/response_key
    #         所以省略对 intent = retrieval_intent/response_key 这种意图生成 rule
    #
    #         rules:
    #           - rule: respond to FAQs
    #             steps:
    #             - intent: faq
    #             - action: utter_faq
    #           - rule: respond to chitchat
    #             steps:
    #             - intent: chitchat
    #             - action: utter_chitchat
    #         """
    #         retrieve_intent = intent.split(RESPONSE_IDENTIFIER_DELIMITER)[0]
    #         action= f"utter_{retrieve_intent}"
    #         intent = retrieve_intent
    #     elif intent not in self.intent_entity_info_mapping:
    #         # 直接根据 意图 做出 action
    #         action = f"utter_{intent}"
    #     elif intent in self.intent_entity_info_mapping and use_custom_actions:
    #         if intent == '查行情':
    #             # 更加 entity 属性做出 action,我们需要自定义 action_inform_stock_info
    #             action = "action_inform_stock_info"
    #         elif intent == '开户咨询':
    #             action = "action_register_account"
    #     elif intent in self.intent_entity_info_mapping and not use_custom_actions:
    #         action = f"utter_{intent}"
    #     return intent, action, is_intent_a_retrieve_intent

    def _make_actionV2(self, intent, use_custom_actions=False) -> [Action, bool]:

        from rasa.nlu.constants import RESPONSE_IDENTIFIER_DELIMITER

        is_intent_a_retrieve_intent = is_rasa_retrieve_intent(intent)
        if is_intent_a_retrieve_intent:
            """
            rasa retrieve_intent: 我们构建 response = f"utter_{intent}/{sub_intent}"
            参考 https://rasa.com/docs/rasa/chitchat-faqs
            对于 retrieval_intent ,我们只需要对 retrival intent 生成 rule
            因为  self.intents 包括 retrieval_intents + retrieval_intent/response_key
            所以省略对 intent = retrieval_intent/response_key 这种意图生成 rule

            rules:
              - rule: respond to FAQs
                steps:
                - intent: faq
                - action: utter_faq
              - rule: respond to chitchat
                steps:
                - intent: chitchat
                - action: utter_chitchat
            """
            retrieve_intent_key = intent.split(RESPONSE_IDENTIFIER_DELIMITER)[-1]
            action_name = f"utter_{intent}"
            action = Action(action_name=action_name, response_text=retrieve_intent_key)
        else:
            # 直接根据 意图 做出 action
            action_name = f"utter_{intent}"
            action = Action(action_name=action_name, response_text=intent)
        return action, is_intent_a_retrieve_intent

    def _transform_raw_intent_data_to_rules_data(self) -> [List[Any], Dict[Text, List[Dict[Text, Text]]]]:
        """
        返回 rules_data 和 actions
        """
        raise NotImplementedError

    def _transform_raw_intent_data_to_stories_data(self):
        raise NotImplementedError


class IVRToRasaDataset(IntentDataToRasaDataset):
    """
    对 IVR 数据转换为 rasa dataset
    """
    BLOCK_ENTITY = 'block'
    STOCK_ENTITY = 'stock'
    ZSSTOCK_ENTITY = 'zsstock'

    # FUNCTION_NAME_TO_INTENT_DICT = {
    #     # 映射到 retrieval_intent 进行识别
    #     # '科创板开通': {'new_label': '开户咨询/科创板开通'},
    #     # '创业板开通': {'new_label': '开户咨询/创业板开通'},
    #     # '查科创板权限': {'new_label': '查看权限/查科创板权限'},
    #     # '查创业板权限': {'new_label': '查看权限/查创业板权限'},
    #
    #     # 映射到 开户咨询，然后通过 entity_values 来进行判断
    #     '科创板开通': {'intent': '开户咨询', 'entity': BLOCK_ENTITY, "entity_values": ['科创板']},
    #     '创业板开通': {'intent': '开户咨询', 'entity': BLOCK_ENTITY, "entity_values": ['创业板']},
    #
    #     '查个股行情': {'intent': '查行情', 'entity': STOCK_ENTITY},
    #     '查指数行情': {'intent': '查行情', 'entity': ZSSTOCK_ENTITY},
    #     '手机软件下载': {'intent': '软件下载/手机软件下载'},
    #     '电脑软件下载': {'intent': '软件下载/电脑软件下载'},
    # }

    def __init__(self, function_name_information_dict=None,
                 **kwargs
                 ):
        super(IVRToRasaDataset, self).__init__(**kwargs)
        self.function_name_information_dict = function_name_information_dict

        if kwargs.get('if_transform_to_rasa_data'):
            # 新的 intent 对应的 entity 信息
            self.intent_entity_info = self._get_intent_entity_info_to_response()
            # 通过 RAW_LABEL_TO_NEW_LABEL_DICT 变换 意图标签
            self.transform_to_rasa_data(if_get_core_data=self.if_get_core_data,
                                        if_get_stories_data=self.if_get_stories_data,
                                        if_get_vocabulary_data=self.if_get_vocabulary_data,
                                        if_get_synonym_data=self.if_get_synonym_data)

        # # 选取重点识别的意图并将其转换为 rasa 的 yaml 格式，以方便标注
        # self.financial_terms = FinancialTerms()
        #
        # # 用于标注 entity
        # self.entity_attribute_to_value_dict = {self.BLOCK_ENTITY: self.financial_terms.get_block_synonyms(),
        #                                        self.STOCK_ENTITY: ['海通证券'],
        #                                        self.ZSSTOCK_ENTITY: ["恒生指数"]}=
        # # entities 和 slots
        # self.entities, self.slots = self._get_entities_slots()

    def _get_intent_entity_info_to_response(self) -> IntentEntityInfoToResponse:
        intent_entity_info_to_response = IntentEntityInfoToResponse(
            function_name_information_dict=self.function_name_information_dict)
        intent_entity_info_to_response.build()
        return intent_entity_info_to_response

    def _apply_intent_changing(self, data):
        data = data.copy()

        def change_label(label):
            """
            # 合并意图 + 增加 entity 信息
            # 1. 合并 科创板开通 和 创业板开通 到 开户查询，并且标注 科创板和创业板为的实体属性为 block
            # 2. 合并 查个股行情 和 查指数行情 到 查行情，并且标注 stock 和 zsstock
            # 3. 变换 手机软件下载 和 电脑软件下载 为检索式标签 软件下载/手机软件下载 和 软件下载/电脑软件下载，使用检索模型进行分类
            """
            if label in self.RAW_LABEL_TO_NEW_LABEL_DICT:
                new_label = self.RAW_LABEL_TO_NEW_LABEL_DICT.get(label).get('new_label')
            else:
                new_label = label
            return new_label

        new_labels = data.loc[:, self.sub_intent_column].apply(change_label)
        data.loc[:, self.sub_intent_column] = new_labels

        new_intent_counts = data.loc[:, self.sub_intent_column].value_counts()
        new_intents = new_intent_counts.index.tolist()
        logger.info(f"new_intent_counts:\n{new_intent_counts}")
        return data, new_intents

    def _transform_raw_intent_data_to_rules_data(self,
                                                 section='rule',
                                                 wait_for_user_input=False,
                                                 use_custom_actions=False,
                                                 if_condition=False
                                                 ):
        """
        需要组件：
        policies:
        - ... # Other policies
        - name: RulePolicy

        rules:
        - rule: Say `hello` whenever the user sends a message with intent `greet`
          steps:
          - intent: greet
          - action: utter_greet

        stories:
        - story: story to find a restaurant
          steps:
          - intent: find_restaurant
          - action: restaurant_form
          - action: utter_restaurant_found

        """
        # 为每个意图 和不同的 entity value 生成一个 action，构建回答的逻辑
        rules = []
        actions_dict: Dict[Text:Action] = {}
        for intent in self.intents:
            # 判断是否是 retrieve_intent
            if is_rasa_retrieve_intent(intent):
                # 根据  intent 来构建 action , action_name = f"utter_{intent}"
                action, is_retrieve_intent = self._make_actionV2(intent=intent)
                if action.action_name not in actions_dict:
                    actions_dict.update({action.action_name: action})
                # retrieve_intent 直接根据 回复已经定义的 response，不需要在 rules 或者 stories 中出现
                continue
            elif not use_custom_actions:
                """
                # 如果不是使用 use_custom_actions，  需要 使用 rule condition：
                结合 intent 和 entity value 做出判断的情况，如 开通创业板，
                我们希望根据不同 entity 生产对应的 rule， 但是实验结果：
                1）the prediction of the action 'utter_创业板开通' in rule '创业板开通' is contradicting with rule(s) '科创板开通' which predicted action 'utter_科创板开通'.
                2） check_for_contradictions: false 的时候，发现 entity value 不起作用
                所以我们转换为 custom action 来处理不同 entity value
                """
                # 找到 intent 对应 实体信息
                entity_info_ls = self.intent_entity_info.get_entity_info(intent)

                # 对不同的 entity_value 进行 action
                for entity_info in entity_info_ls:
                    condition = None
                    entity_to_value_list = None
                    entity = entity_info.get(self.intent_entity_info.entity_key)
                    entity_values = entity_info.get(self.intent_entity_info.entity_values_key)
                    # 使用 secondary_intent_key 作为 response text
                    response = entity_info.get(self.intent_entity_info.response_key)

                    if entity is None:
                        action, _ = self._make_actionV2(intent=intent)
                    # intent 是否存在对应的 entities
                    if entity is not None and entity_values is None:
                        # 根据 intent + entity_type 来确定 action: action_name = response_intent
                        action, _ = self._make_actionV2(intent=response)
                        # 对于只需要 根据 entity 属性就判断的情况 ：查个股行情 + 查指数行情
                        entity_to_value_list = [{entity: True}]
                        condition = [{'slot_was_set': entity_to_value_list}]
                    # TODO: 目前只对 意图下 单个 的 entity 进行判断逻辑
                    elif entity_values is not None:
                        # 根据 intent + entity_type + entity_values 来确定 action: action_name = response_intent
                        action, _ = self._make_actionV2(intent=response)
                        # 将识别 的 entity value 值作为 condition
                        entity_to_value_list = [{entity: entity_values[0]}]
                        condition = [{'slot_was_set': entity_to_value_list}]

                    if if_condition:
                        # 将 condition 作为 rule，这样做其实对于 faq 情景下不太合理，因为condition 指的是提前有 slot
                        rule = {"rule": response,
                                "condition": condition,
                                "steps": [{"intent": intent},
                                          {'action': action.action_name}],
                                "wait_for_user_input": wait_for_user_input
                                }
                    else:
                        # 当使用自定义的 action 的时候，传入参数 wait_for_user_input，试图使得 之前的slot值不会影响后面的rule
                        if entity_to_value_list is None:
                            # 当使用  意图进行判断的action的时候
                            rule = {"rule": intent,
                                    "steps": [{'intent': intent},
                                              {'action': action.action_name}],
                                    "wait_for_user_input": wait_for_user_input
                                    }
                        else:
                            # 当 entity信息使用判断的action的时候
                            # https://github.com/rasahq/rasa/issues/8581
                            rule = {"rule": response,
                                    # "condition":condition,
                                    "steps": [{"intent": intent,
                                               "entities": entity_to_value_list},
                                              {"slot_was_set": entity_to_value_list},
                                              {'action': action.action_name}],
                                    "wait_for_user_input": wait_for_user_input
                                    }
                    if action.action_name not in actions_dict:
                        actions_dict.update({action.action_name: action})
                    rules.append(rule)

        actions = [action for action_name, action in actions_dict.items()]
        return rules, actions

    def _transform_raw_intent_data_to_stories_data(self):
        """
        stories:
            - story: collect restaurant booking info  # name of the story - just for debugging
              steps:
              - intent: greet                         # user message with no entities
              - action: utter_ask_howcanhelp
              - intent: inform                        # user message with entities
                entities:
                - location: "rome"
                - price: "cheap"
              - action: utter_on_it                  # action that the bot should execute
              - action: utter_ask_cuisine
              - intent: inform
                entities:
                - cuisine: "spanish"
              - action: utter_ask_num_people
        """

        # 为每个意图 和不同的 entity value 生成一个 action，构建回答的逻辑
        stories = []
        for intent in self.intents:
            # 直接根据 意图 做出 action
            action = self._make_action(intent)
            story = {"story": intent, "steps": [{'intent': intent}, {'action': action}]}
            stories.append(story)
            if intent in self.intent_entity_info_mapping:
                entity_info_ls = self.intent_entity_info_mapping.get(intent)
                # 对不同的 entity_value 进行 action
                for entity_info in entity_info_ls:
                    raw_label = entity_info.get('raw_label')
                    entity = entity_info.get('entity')
                    entity_values = entity_info.get("entity_values")
                    action = self._make_action(intent, raw_label, entity, entity_values)
                    # intent 是否存在对应的 entities
                    # TODO: 目前只对 意图下 单个 的 entity 进行判断逻辑
                    if entity_values is not None and entity_values.__len__() == 1:
                        nlu_output = {'intent': intent, 'entities': [{entity: entity_values[0]}]}
                    else:
                        # 对于需要只需要 更加 entity 属性就判断的情况 ：查个股行情 + 查指数行情
                        # TODO: 如何处理？
                        # 参考 ： https://stackoverflow.com/questions/67069899/problem-with-stories-decided-by-entities-using-rasa
                        nlu_output = {'intent': intent, 'entities': [entity]}

                    story = {"story": raw_label, "steps": [nlu_output, {'action': action}]}
                    stories.append(story)
        return stories


# def transform_hsnlp_faq_data_to_rasa_nlu_format(standard_to_extend_question_dict):
#     rasa_nlu_data_ls = []
#     for standard_id, standard_id_dict in standard_to_extend_question_dict.items():
#         standard_questions_ls = standard_id_dict.get('standard_questions')
#         assert standard_questions_ls.__len__() == 1
#         # 将标准问转换为 retrive intent
#         standard_question = standard_questions_ls[0].get('question')
#         # 消除标点符号
#         standard_indent = "".join(
#             [char for char in standard_question if not is_punctuation(char)])
#         intent = f"faq/{standard_indent}"
#         extend_questions_ls = standard_id_dict.get('extend_questions')
#         examples = []
#         examples.append(standard_question)
#         for extend_question_info in extend_questions_ls:
#             examples.append(extend_question_info.get('question'))
#         # 将 examples 组装为 一个 str：
#         # - example\n- example\n
#         examples_str = "".join([f"- {example}\n" for example in examples])
#         # 使用  PreservedScalarString: https://mlog.club/article/1462558
#         retrieval_data = {"intent": intent, "examples": PSS(examples_str)}
#         rasa_nlu_data_ls.append(retrieval_data)
#     return rasa_nlu_data_ls
#
#
# def transform_raw_intent_data_rasa_nlu_and_response_data(raw_intent_data, use_retrieve_intent=True,
#                                                          intent_column=INTENT,
#                                                          sub_intent_column=SUB_INTENT,
#                                                          use_intent_column_as_label=True):
#     """
#     use_retrieve_intent:  使用  intent = f"{intent}/{sub_intent}" 来构建 rasa retrieve_intent
#     use_intent_column_as_label: Ture 的时候使用 intent 作为 label ,False 的时候使用 sub_intent 作为 label
#     """
#     rasa_nlu_data_ls = []
#     response_data_ls = []
#     nlu_example_num = 0
#     if use_retrieve_intent:
#         sub_intent_data = raw_intent_data.loc[raw_intent_data.loc[:, sub_intent_column].notna()].copy()
#         none_sub_intent_data = raw_intent_data.loc[raw_intent_data.loc[:, sub_intent_column].isna()].copy()
#         assert sub_intent_data.shape[0] + none_sub_intent_data.shape[0] == raw_intent_data.shape[0]
#
#         # 对于具有子意图的数据
#         grouped_sub_intent_data = sub_intent_data.groupby(by=[intent_column, sub_intent_column])
#         for (intent, sub_intent), group in grouped_sub_intent_data:
#             examples = group.loc[:, 'question'].tolist()
#             intent = f"{intent}/{sub_intent}"
#
#             examples_str = "".join([f"- {example}\n" for example in examples])
#             # 使用  PreservedScalarString: https://mlog.club/article/1462558
#             nlu_data = {"intent": intent, "examples": PSS(examples_str)}
#             rasa_nlu_data_ls.append(nlu_data)
#             # 获取 response
#             response = group.iloc[0].loc['response']
#             response_data_ls.append({"intent": intent, "response": response})
#             nlu_example_num += examples.__len__()
#
#         # 对于没有子意图的数据
#         grouped_none_sub_intent_data = none_sub_intent_data.groupby(by=['intent'])
#         for intent, group in grouped_none_sub_intent_data:
#             examples = group.loc[:, 'question'].tolist()
#             intent = f"{intent}"
#             examples_str = "".join([f"- {example}\n" for example in examples])
#             # 使用  PreservedScalarString: https://mlog.club/article/1462558
#             intent_classifier_data = {"intent": intent, "examples": PSS(examples_str)}
#             rasa_nlu_data_ls.append(intent_classifier_data)
#             nlu_example_num += examples.__len__()
#
#     else:
#         if use_intent_column_as_label:
#             label_column = intent_column
#         else:
#             label_column = sub_intent_column
#
#         # 对于所有的数据按照意图分组
#         grouped_none_sub_intent_data = raw_intent_data.groupby(by=label_column)
#         for intent, group in grouped_none_sub_intent_data:
#             examples = group.loc[:, 'question'].tolist()
#             intent = f"{intent}"
#             examples_str = "".join([f"- {example}\n" for example in examples])
#             # 使用  PreservedScalarString: https://mlog.club/article/1462558
#             intent_classifier_data = {"intent": intent, "examples": PSS(examples_str)}
#
#             # 获取 response
#             response = group.iloc[0].loc['response']
#             response_data_ls.append({"intent": intent, "response": response})
#             rasa_nlu_data_ls.append(intent_classifier_data)
#             nlu_example_num += examples.__len__()
#     assert nlu_example_num == raw_intent_data.shape[0]
#
#     # 输出 responses
#     rasa_responses_data_dict = {}
#     for response_data in response_data_ls:
#         intent = response_data.get('intent')
#         utter_intent = f"utter_{intent}"
#
#         response = response_data.get('response')
#         if pd.isna(response):
#             response = f"{intent}+response"
#         rasa_responses_data_dict.update({utter_intent: [{'text': response}]})
#
#     return rasa_nlu_data_ls, rasa_responses_data_dict
#
#
# def transform_to_rasa_nlu_data(data,
#                                dataset='standard_to_extend_question_dict',
#                                path=None, topK=None,
#                                use_retrieve_intent=True,
#                                use_intent_column_as_label=True):
#     """
#     转换 standard_to_extend_question_dict 为 rasa_nlu_data 数据格式
#     nlu:
#       - intent: chitchat/ask_name
#         examples: |
#           - What is your name?
#           - May I know your name?
#           - What do people call you?
#           - Do you have a name for yourself?
#       - intent: chitchat/ask_weather
#         examples: |
#           - What's the weather like today?
#           - Does it look sunny outside today?
#           - Oh, do you mind checking the weather for me please?
#           - I like sunny days in Berlin.
#     """
#     if dataset == 'standard_to_extend_question_dict':
#         rasa_nlu_data_ls, = transform_hsnlp_faq_data_to_rasa_nlu_format(data)
#     elif dataset in ['intent_classification', 'ivr_data']:
#         rasa_nlu_data_ls, rasa_responses_data_ls = \
#             transform_raw_intent_data_rasa_nlu_and_response_data(data, use_retrieve_intent=use_retrieve_intent,
#                                                                  use_intent_column_as_label=use_intent_column_as_label)
#
#     # 只选取 前topk 做实验
#     if topK:
#         rasa_nlu_data_ls = rasa_nlu_data_ls[:topK]
#     rasa_nlu_data = {"version": "2.0",
#                      "nlu": rasa_nlu_data_ls}
#     write_to_yaml(data=rasa_nlu_data, path=path)
#     logger.info(f"保存 rasa_nlu_data 共 {data.shape[0]} 条到 {path}")
#
#     # 增加 responses
#     if rasa_responses_data_ls:
#         rasa_responses_data = {"version": "2.0",
#                                "responses": rasa_responses_data_ls}
#         responses_path = os.path.join(os.path.dirname(path), 'responses.yml')
#         write_to_yaml(data=rasa_responses_data, path=responses_path)
#         logger.info(f"保存 rasa_responses_data 到 {responses_path}")
#     return rasa_nlu_data

class HSfaqToRasaDataset(IVRToRasaDataset):
    def __init__(self, entity_data: EntityData,
                 standard_question_knowledge_data: StandardQuestionKnowledgeData,
                 **kwargs):
        super(HSfaqToRasaDataset, self).__init__(**kwargs)
        self.entity_data = entity_data
        self.standard_question_knowledge_data = standard_question_knowledge_data

    def _transform_raw_intent_data_to_rules_data(self):
        """
        重写 该方法，直接使用 StandardQuestionKnowledgeData 对象 生成的 rule 和 actions
        """
        self.standard_question_knowledge_data.get_actions_and_rules()
        rules = self.standard_question_knowledge_data._rules
        actions = self.standard_question_knowledge_data._actions
        return rules, actions

    def _transform_raw_intent_data_to_rasa_nlu_data(self, raw_intent_data,
                                                    intent_column=INTENT,
                                                    sub_intent_column=SUB_INTENT,
                                                    question_column=QUESTION,
                                                    use_retrieve_intent=False,
                                                    use_intent_column_as_label=True):
        """
        use_retrieve_intent:  使用  intent = f"{intent}/{sub_intent}" 来构建 rasa retrieve_intent
        use_intent_column_as_label: Ture 的时候使用 intent 作为 label ,False 的时候使用 sub_intent 作为 label

        增加 attribute 到数据中
        """
        rasa_nlu_data_ls = []
        intents = set()
        # attributes = set()
        nlu_example_num = 0

        intent_to_attribute_mapping = defaultdict(set)
        # 对于所有的数据按照意图 + attribute 分组
        grouped_intent_data = raw_intent_data.groupby(by=[INTENT, ATTRIBUTE])
        for group_key, group_index in grouped_intent_data.groups.items():
            # 对不同的意图下根据 attribute 进行分组
            intent, attribute = group_key
            group = raw_intent_data.loc[group_index]
            examples = group.loc[:, question_column].tolist()
            key_value_dict = {INTENT: intent}
            if not pd.isna(attribute):
                key_value_dict.update({ATTRIBUTE: attribute})
            intent_attribute_classifier_data = _to_yaml_string_dict(key_value_dict=key_value_dict, examples=examples)
            rasa_nlu_data_ls.append(intent_attribute_classifier_data)
            intents.add(intent)
            # attributes.add(attribute)
            intent_to_attribute_mapping[intent].add(attribute)

            nlu_example_num += examples.__len__()

        assert nlu_example_num == raw_intent_data.shape[0]
        intents = list(intents)
        intent_to_attribute_mapping ={k:list(v) for k,v in intent_to_attribute_mapping.items()}
        attributes = []
        for intent,attribute_ls  in intent_to_attribute_mapping.items():
            attributes.extend(attribute_ls)

        return rasa_nlu_data_ls, intents,intent_to_attribute_mapping


class DiagnosisDataToRasaDataset(IntentDataToRasaDataset):

    def _get_entities_slots(self):
        entity_types = []
        slots = dict()
        for entity_type, values in self.entity_attribute_to_value_dict.items():
            # 去掉 entity_type 末尾的复数s
            entity_type = entity_type.lower()
            entity_types.append(entity_type)
            if entity_type in ['department']:
                slot_type = "categorical"
                slots.update({entity_type: {"type": slot_type, "values": list(values)}})
            elif entity_type in ['disease', 'symptom', 'check', 'drug', 'food', 'producer', 'recipe']:
                slot_type = "text"
                slots.update({entity_type: {"type": slot_type}})
        return entity_types, slots
