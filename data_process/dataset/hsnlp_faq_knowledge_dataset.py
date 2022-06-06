#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/12/1 16:17 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/1 16:17   wangfc      1.0         None
"""
import os
from pathlib import Path
from typing import Union, Text, List, Dict, Any, Tuple, NamedTuple
import re
from collections import defaultdict, namedtuple
from collections import OrderedDict
import pandas as pd

from data_process.data_utils import make_rasa_action, Action
from data_process.dataset.finacial_knowledge_base import FinancialTerms
from utils.constants import INTENT, STANDARD_QUESTION, ENTITIES, RESPONSE
from models.trie_tree import TrieTree
from utils.io import _to_yaml_examples_dict, dataframe_to_file
import logging

logger = logging.getLogger(__name__)
DEFAULT_INTENT = '其他'
# FIRST_KNOWLEDGE_ZH = '一级知识点'
INTENT_ZH = '意图'
ATTRIBUTE_ZH = '属性'
# FIRST_KNOWLEDGE = 'first_knowledge'
# SECOND_KNOWLEDGE = 'second_knowledge'
ATTRIBUTE = 'attribute'
DEFAULT_ATTRIBUTE = '其他'
# FIRST_INTENT = 'first_intent'
# SECONDARY_INTENT = "secondary_intent"  # 作为 intent + entity 返回的意图
ENTITY = 'entity'
ENTITY_TYPE = 'entity_type'
ENTITY_TYPES = 'entity_types'
ENTITY_VALUES = 'entity_values'
ENTITY_VALUE = 'entity_value'
ENTITY_START = 'entity_start'
ENTITY_END = 'entity_end'

ENTITY_TYPE_PATTERN = f"{ENTITY_TYPE}_(\d)"
ENTITY_VALUE_PATTERN = f"{ENTITY_VALUE}_(\d)"

DEFAULT_RESPONSE = ''

INTENT_ATTRIBUTE_SPLIT_CHAR = '_'
KNOWLEDGE_COMBINE_CHAR = "/"
ATTRIBUTE_SPLIT_CHAR = '/'


class EntityData(FinancialTerms):
    """
    创建一个 entity的对象
    """

    def __init__(self, data_path=None, output_dir=None):
        super(EntityData, self).__init__(financial_term_data_path=data_path)

        self.text_slot_types = ['company_name', 'stock', 'zsstock', 'time', 'location']
        self._financial_entity_data = self.get_financial_term_data()

        # 所有的 entity_types

        self._entity_type_zh_to_en_mapping = self._get_entity_type_zh_to_en_mapping()
        # self._entity_types = self._get_key_values(key='entity_types')
        self._entity_types = [en for zh, en in self._entity_type_zh_to_en_mapping.items()]
        logger.info(f"实体属性共有 {self._entity_types.__len__()} 个:{self._entity_types}")

        # entity_type 对应的值
        self._entity_type_to_value_dict = self._get_entity_type_to_value_dict()

        # 实体的近义词
        # self.synonyms_dict = {}

        # 所有的 slots
        self._slots = self._get_slots()

        # 创建 entity_tire_tree，可以用于对原始数据进行标注
        self._entity_tire_tree = self.build_entity_tire_tree(
            entity_attribute_to_value_dict=self._entity_type_to_value_dict)

    def _get_key_values(self, key='entity_type') -> List[Text]:
        """
        获取 yml 数据 指定 key 对应的数据
        """
        values = []
        for d in self._financial_entity_data:
            data_type = self._get_data_type(data=d)
            data_key = d[data_type]
            if data_key == key:
                examples = d[self.examples_key]
                values.extend(examples)
        if values and isinstance(values[0], str):
            values = list(set(values))
        return values

    def _get_entity_type_zh_to_en_mapping(self, key="entity_type_en_to_zh_mapping"):
        entity_type_en_to_zh_mapping_ls = self._get_key_values(key=key)
        entity_type_en_to_zh_mapping = {}
        for en_to_zh_dict in entity_type_en_to_zh_mapping_ls:
            entity_type_en_to_zh_mapping.update(en_to_zh_dict)
        return {zh.strip(): en.strip() for en, zh in entity_type_en_to_zh_mapping.items()}

    def _get_data_type(self, data: Dict[Text, Any]):
        """
        获取  ymal 文件中每个 字段数据的 数据类型： lookup, synonym, regex,
        """
        data_type = list(set(data.keys()).difference({self.examples_key}))[0]
        return data_type

    def _get_entity_type_to_value_dict(self):
        entity_type_to_values_dict = {}
        for entity_type in self._entity_types:
            standard_entity_values = self._get_key_values(key=entity_type)
            # 获取 value 对应的同义词
            all_entity_values = standard_entity_values.copy()
            for entity_value in standard_entity_values:
                entity_value_synonyms = self._get_key_values(key=entity_value)
                all_entity_values.extend(entity_value_synonyms)
            entity_type_to_values_dict.update({entity_type: sorted(list(set(all_entity_values)), key=lambda x: len(x))})
            logger.debug(f"实体类型:{entity_type},标准的实体值={standard_entity_values},扩展同义词后的实体值={all_entity_values}")
        return entity_type_to_values_dict

    def _get_slots(self) -> Dict[Text, Dict[Text, Text]]:
        """
        https://stackoverflow.com/questions/67069899/problem-with-stories-decided-by-entities-using-rasa
        对 STOCK_ENTITY，ZSSTOCK_ENTITY 只是进行entity属性（而不是基于entity value）的判读做出 action，
        应该设置 该 slot 为 categorical，
        在 stories.yml 中只对 entity 属性 进行编辑

        slot type ： text，bool, float, categorical, list,any
        可以用于存储各种信息类型，用于对话管理
        """
        # entities = []
        slots = dict()
        for entity_type, values in self._entity_type_to_value_dict.items():
            # entities.append(entity_type)
            if entity_type in self.text_slot_types:
                slot_type = "text"
                slots.update({entity_type: {"type": slot_type}})
            else:
                # if entity_type in ['security_market', 'stock_block', 'password_type', 'account_type',
                #                    'bussiness_type', 'fee_type', 'operation_channel', 'nationality']:
                slot_type = "categorical"
                slots.update({entity_type: {"type": slot_type, "values": values}})
        return slots

    @staticmethod
    def build_entity_tire_tree(entity_attribute_to_value_dict: Dict[Text, List[Text]]) -> TrieTree:
        """
        使用 entity_attribute_to_value_dict 建立一个 entity_tire_tree

        """
        entity_tire_tree = TrieTree()
        for entity_type, entities in entity_attribute_to_value_dict.items():
            for entity in entities:
                entity_tire_tree.insert(sequence=entity, word_type=entity_type)
        logger.info(f"建立了 entity_tire_tree 共 {entity_tire_tree.node_num} 个节点")
        # entity_tire_tree.get_nodes_start_with_prefix('牙疼')
        return entity_tire_tree

    def _build_entity_synonym_data(self, data_types=["lookup", "synonym"]):
        """
        获取实体的近义词数据
        """
        entity_synonym_data = self._get_data_type_to_values(data_types=data_types)
        return entity_synonym_data

    def _build_entity_regex_data(self, data_types=["regex"]):
        entity_regex_data = self._get_data_type_to_values(data_types=data_types)
        return entity_regex_data

    def _get_data_type_to_values(self, data_types: List[Text], exclude_value=["entity_type_en_to_zh_mapping"]) \
            -> List[Dict[Text, List[Text]]]:
        """
        获取指定 data_types 中的数据
        """
        # financial_term_data = self.financial_terms.financial_terms_data
        synonym_data_ls = []
        for financial_term_dict in self._financial_entity_data:
            # data_type 包括： lookup, synonym, regex
            data_type = self._get_data_type(data=financial_term_dict)
            value = financial_term_dict.get(data_type)
            if data_type in data_types and value not in exclude_value:
                examples = financial_term_dict.get(self.examples_key)
                assert isinstance(examples, list)
                yaml_examples_dict = _to_yaml_examples_dict(data_type, value, examples)
                synonym_data_ls.append(yaml_examples_dict)
        return synonym_data_ls

    def _update(self, entity_type_to_value_dict: Dict[Text, List[Text]]):

        for key, value in entity_type_to_value_dict.items():
            # 去除 不在 entity_type_zh 中的key
            entity_type = self._entity_type_zh_to_en_mapping.get(key)
            if entity_type is None:
                # entity_type_to_value_dict.pop(key)
                pass
            else:
                # 转换为 英文的key
                entity_values_new = entity_type_to_value_dict.get(key)
                # entity_type 对应的值
                entity_values_original = self._entity_type_to_value_dict.get(entity_type)
                entity_value_add = sorted(set(entity_values_new).difference(entity_values_original),
                                          key=lambda x: len(x))
                logger.info(f"entity_type={entity_type},\n"
                            f"entity_types_original={entity_values_original},\n"
                            f"entity_value_add={entity_value_add}")
                entity_value_union = sorted(set(entity_values_new).union(entity_values_original),
                                            key=lambda x: len(x))
                self._entity_type_to_value_dict.update({entity_type: entity_value_union})
        # 所有的 slots
        self._slots = self._get_slots()

        # 创建 entity_tire_tree，可以用于对原始数据进行标注
        self._entity_tire_tree = self.build_entity_tire_tree(
            entity_attribute_to_value_dict=self._entity_type_to_value_dict)


class KnowledgeDefinitionData():
    """
    @time:  2021/12/2 8:44
    @author:wangfc
    @version:
    @description: 读取 知识库的定义信息：
    意图，属性，实体类型，实体值

    @params:
    @return:
    """

    def __init__(self, raw_knowledge_definition_file_path,
                 intent_attribute_definition_sheet_name="意图和属性定义-20220207",
                 entity_definition_sheet_name="实体定义-20220207",
                 intent_attribute_definition_path=None,
                 entity_definition_path=None,
                 ):
        """
        原来的一个 sheet 转换为意图属性 和 实体的定义分开为2个 sheet
        """
        self.raw_knowledge_definition_file_path = raw_knowledge_definition_file_path
        self.intent_attribute_definition_sheet_name = intent_attribute_definition_sheet_name
        self.entity_definition_sheet_name = entity_definition_sheet_name

        self.intent_attribute_definition_path = intent_attribute_definition_path
        self.entity_definition_path = entity_definition_path

        if intent_attribute_definition_path is None or not os.path.exists(intent_attribute_definition_path):
            self.intent_attribute_definition_table = dataframe_to_file(path=self.raw_knowledge_definition_file_path,
                                                                       mode='r',
                                                                       sheet_name=intent_attribute_definition_sheet_name)
        else:
            self.intent_attribute_definition_table = dataframe_to_file(path=self.intent_attribute_definition_table)
        if entity_definition_path is None or not os.path.exists(entity_definition_path):
            self.entity_definition_table = dataframe_to_file(path=self.raw_knowledge_definition_file_path, mode='r',
                                                             sheet_name=entity_definition_sheet_name)
        else:
            self.entity_definition_table = dataframe_to_file(path=self.entity_definition_path)

        # self._first_knowledge = self._get_first_knowledge()
        self._intents = self._get_intents()
        self._attributes = self._get_attributes()
        self._entity_types = self._get_entity_types()
        self._entity_type_to_value_dict = self._get_entity_type_to_value_dict()
        logger.info(f"faq知识定义中的意图共{self._intents.__len__()}个：{self._intents}\n"
                    f"faq知识定义中的属性共{self._attributes.__len__()}个：{self._attributes}\n"
                    f"faq知识定义中的实体类型共{self._entity_types.__len__()}个：{self._entity_types}")

    # def _get_first_knowledge(self) -> List[Text]:
    #     first_knowledge = self.knowledge_definition_table.loc[:, FIRST_KNOWLEDGE_ZH] \
    #         .dropna().drop_duplicates().apply(str.strip).tolist()
    #     return first_knowledge

    def _get_intents(self) -> List[Text]:
        intents = self.intent_attribute_definition_table.loc[:, INTENT_ZH] \
            .dropna().drop_duplicates().apply(str.strip).tolist()
        return intents

    def _get_attributes(self) -> List[Text]:
        attributes = self.intent_attribute_definition_table.loc[:, ATTRIBUTE_ZH] \
            .dropna().drop_duplicates().apply(str.strip).tolist()
        return attributes

    def _get_entity_types(self) -> List[Text]:
        entity_types = self.entity_definition_table.loc[:, ENTITY_TYPES].dropna().drop_duplicates() \
            .apply(str.strip).tolist()
        return entity_types

    def _get_entity_type_to_value_dict(self) -> Dict[Text, Text]:
        entity_type_to_value_df = self.entity_definition_table.loc[:, [ENTITY_TYPES, ENTITY_VALUES]] \
            .dropna().drop_duplicates().astype(str)
        entity_type_to_value_dict = {}
        for i in range(entity_type_to_value_df.__len__()):
            row = entity_type_to_value_df.iloc[i]
            entity_type = row.loc[ENTITY_TYPES]
            entity_values = [e for e in re.split(pattern=r",|，|\n|\\n", string=row.loc[ENTITY_VALUES])
                             if re.match(pattern='\w+', string=e)]
            entity_type_to_value_dict.update({entity_type: entity_values})
        return entity_type_to_value_dict


class StandardQuestionInfo():
    """
    标准问对应的信息
    depreciated :  使用 QuestionKnowledge 代替
    """

    def __init__(self, value: Dict):
        self.intent_column = INTENT
        self.vocabulary_key = "vocabulary"
        self.data = value

    def get_vocabulary(self, filter_intents: Union[List[Text], Dict[Text, int]] = None):
        """
        从 self.function_name_to_code_dict 中获取 vocabulary
        """
        vocabulary = set()
        for function_name, function_name_info in self.data.items():
            intent = function_name_info.get(self.intent_column)
            if filter_intents is None or \
                    (filter_intents is not None and intent is not None and intent in filter_intents):
                function_name_vocabulary = function_name_info.get(self.vocabulary_key)
                if function_name_vocabulary is not None:
                    words = [word_and_counts.split('|')[0].lower() for word_and_counts in
                             function_name_vocabulary.split(',')]
                    vocabulary = vocabulary.union(set(words))
        vocabulary = sorted(list(vocabulary), key=lambda x: (len(x), x))
        return vocabulary

    def get(self, key):
        return self.data.get(key)


class QuestionKnowledge():
    """
    记录一个 question,及其对应的知识信息
    """
    # 问题对应的知识的key
    # KNOWLEDGE_KEYS = [FIRST_KNOWLEDGE, ATTRIBUTE, INTENT, ENTITIES]
    KNOWLEDGE_KEYS = [INTENT, ATTRIBUTE, ENTITIES]
    KNOWLEDGE_KEY_VALUE_JOINED_CHAR = '-'
    KNOWLEDGE_COMBINED_CHAR = '/'

    def __init__(self,
                 knowledge_dict: Dict[Text, Text],
                 entity_type_zh_to_en_mapping: Dict[Text, Text],
                 question: Text = None,
                 ):
        self._question = question
        self._knowledge_dict = OrderedDict(knowledge_dict)
        # 将人工标注的 entity 转换为 英文
        self._entity_type_zh_to_en_mapping = entity_type_zh_to_en_mapping
        self._knowledge_as_string = self._knowledge_dict_to_string()

        # self._first_knowledge = self._get_first_knowledge()
        self._attribute = self._get_attribute()
        self._intent = self._get_intent()

        self._entities = self._get_entities()
        self._entity_num = self._entities.__len__()

    def get(self, key: Text):
        # 如果key 存在，则获取对应的属性
        if key in self.KNOWLEDGE_KEYS:
            attribute_name = f"_{key}"
            return getattr(self, attribute_name)

    def __getitem__(self, item:Text):
        if item in self.KNOWLEDGE_KEYS:
            attribute_name = f"_{item}"
            return getattr(self, attribute_name)

    # def _get_first_knowledge(self) -> Text:
    #     return self._knowledge_dict[FIRST_KNOWLEDGE_ZH]

    def _get_attribute(self) -> Text:
        return self._knowledge_dict.get(ATTRIBUTE_ZH)

    # def _get_intent(self) -> Text:
    #     """
    #     目前，我们使用 first_knowledge + attribute = intent
    #     """
    #     if self._attribute is None:
    #         intent = f"{self._first_knowledge}"
    #     else:
    #         intent = f"{self._first_knowledge}_{self._attribute}"
    #     return intent

    def _get_intent(self, with_attribute=False) -> Text:
        """
        目前，我们使用 first_knowledge + attribute = intent
        """
        intent = self._knowledge_dict.get(INTENT_ZH)
        if with_attribute and self._attribute:
            intent = f"{intent}{INTENT_ATTRIBUTE_SPLIT_CHAR}{self._attribute}"
        return intent

    def _get_entities(self) -> List[Dict[Text, Text]]:
        """
        将 entity_type 转换为 英文
        """
        entity_type_to_value_list = []
        for key, entity_type_zh in self._knowledge_dict.items():
            matched = re.match(pattern=ENTITY_TYPE_PATTERN, string=key)
            if matched:
                entity_index = matched.groups()[0]
                entity_type_key = key
                if self._entity_type_zh_to_en_mapping:
                    entity_type = self._entity_type_zh_to_en_mapping[entity_type_zh]
                else:
                    # 当没有映射关系的时候，使用中文作为 entity_type
                    entity_type = entity_type_zh
                entity_value_key = f"{ENTITY_VALUE}_{entity_index}"
                entity_value = self._knowledge_dict[entity_value_key]
                entity_type_to_value_list.append({entity_type: entity_value})
        return entity_type_to_value_list

    def _knowledge_dict_to_string(self) -> Text:
        key_value_ls = [f"{key}{self.KNOWLEDGE_KEY_VALUE_JOINED_CHAR}{value}" for key, value in
                        self._knowledge_dict.items()]
        knowledge_str = f"{self.KNOWLEDGE_COMBINED_CHAR}".join(key_value_ls)
        return knowledge_str

    @classmethod
    def _knowledge_dict_str_to_dict(cls, knowledge_dict_str):
        knowledge_dict = {}
        knowledge_part_ls = knowledge_dict_str.split(cls.KNOWLEDGE_COMBINED_CHAR)
        for knowledge_part in knowledge_part_ls:
            key_value = knowledge_part.split(cls.KNOWLEDGE_KEY_VALUE_JOINED_CHAR)
            assert key_value.__len__() == 2
            key, value = key_value
            knowledge_dict.update({key: value})
        return knowledge_dict

    @classmethod
    def instance_from_knowledge_dict_str(cls, knowledge_dict_str, entity_type_zh_to_en_mapping):
        """
        从  knowledge_dict_str 返回为 QuestionKnowledge 对象
        """
        knowledge_dict = cls._knowledge_dict_str_to_dict(knowledge_dict_str)
        question_knowledge = cls(knowledge_dict=knowledge_dict,
                                 entity_type_zh_to_en_mapping=entity_type_zh_to_en_mapping)
        return question_knowledge


class StandardQuestionKnowledgeData():
    """
    @time:  2021/12/1 16:09
    @author:wangfc
    @version:
    @description: 标准问的知识信息

    @params:
    @return:
    """

    def __init__(self, standard_question_knowledge_file_path: Union[Text, Path],
                 sheet_name="haitong-ivr标准问",
                 output_dir=None, entity_type_zh_to_en_mapping: Dict[Text, Text] = None):

        self._standard_question_knowledge_file_path = standard_question_knowledge_file_path
        self._sheet_name = sheet_name
        self._output_dir = output_dir
        self._entity_type_zh_to_en_mapping = entity_type_zh_to_en_mapping

        # 读取标准问的定义表格
        self._standard_question_knowledge_table = self._read_standard_question_knowledge_data(
            self._standard_question_knowledge_file_path, self._sheet_name)

        # 标准问对应知识：  允许一个 standard_question 对应多个 知识
        # TODO: 分割 standard_question 对应多个属性的情况
        self._standard_question_to_knowledge_ls = self._get_standard_question_to_knowledge_ls(
            self._standard_question_knowledge_table, self._knowledge_keys)

        # 所有的标准问
        self._standard_questions = list(self._standard_question_to_knowledge_ls.keys())

        # # 获取所有的 first_knowledge
        # self._first_knowledge = self._standard_question_knowledge_table.loc[:,
        #                         FIRST_KNOWLEDGE_ZH].dropna().drop_duplicates().apply(str.strip).tolist()
        #
        # # 获取所有的 attribute
        # self._attributes = self._standard_question_knowledge_table.loc[:,
        #                    ATTRIBUTE_ZH].dropna().drop_duplicates().apply(str.strip).tolist()

        # self._first_knowledge = self._get_first_knowledge()
        self._intents = self._get_intents()
        self._attributes = self._get_attributes()

        # 知识 对应所有标准问：不同知识可能对应同一个标准问
        self._knowledge_to_standard_questions = self._get_knowledge_to_standard_questions()

        # 获取可以使用知识唯一确定的标准问和不能的
        # 区分可以唯一确定的标注问 vs 不能唯一确定的标注问 ：判断哪些标准问是可以唯一确定的
        self._knowledge_based_standard_questions, self._none_knowledge_based_standard_questions = \
            self._get_knowledge_based_questions()
        self._knowledge_based_standard_questions_ratio = self._get_knowledge_based_standard_questions_ratio()
        # 获取所有的意图
        self._intent_to_standard_questions = self._get_intent_to_standard_questions()

        self.get_actions_and_rules()

    def _read_standard_question_knowledge_data(self, path, sheet_name):
        standard_question_knowledge_table = dataframe_to_file(path, sheet_name=sheet_name, mode='r')

        select_columns = self._select_columns(standard_question_knowledge_table.columns.tolist())

        standard_question_knowledge_table = standard_question_knowledge_table.loc[:, select_columns]

        # 去掉 一级知识点为空的数据
        # standard_question_knowledge_table.dropna(subset=[FIRST_KNOWLEDGE_ZH], inplace=True)
        standard_question_knowledge_table.dropna(subset=[INTENT_ZH], inplace=True)
        self._knowledge_keys = select_columns[1:]
        return standard_question_knowledge_table

    def _select_columns(self, columns):
        selected_columns = []
        for column in columns:
            # if column in [STANDARD_QUESTION, FIRST_KNOWLEDGE_ZH, ATTRIBUTE_ZH]:
            if column in [STANDARD_QUESTION, INTENT_ZH, ATTRIBUTE_ZH]:
                selected_columns.append(column)
            elif re.match(pattern=ENTITY_TYPE_PATTERN, string=column) or \
                    re.match(pattern=ENTITY_VALUE_PATTERN, string=column):
                selected_columns.append(column)
        return selected_columns

    def _get_standard_question_to_knowledge_ls(self, standard_question_knowledge_table, knowledge_keys) \
            -> Dict[Text, List["QuestionKnowledge"]]:
        """
        标准问对应知识：  允许一个 standard_question 对应多个 知识
        """
        standard_question_to_knowledge_ls = {}
        for i in range(standard_question_knowledge_table.__len__()):
            standard_question = self._standard_question_knowledge_table.iloc[i].loc[STANDARD_QUESTION]
            raw_knowledge_dict = self._standard_question_knowledge_table.iloc[i].loc[knowledge_keys].dropna().to_dict()
            #  TODO: 分割 standard_question 对应多个属性的情况
            attribute_value = raw_knowledge_dict.get(ATTRIBUTE_ZH)
            knowledge_dict_ls = []
            if isinstance(attribute_value, str) and attribute_value.split(ATTRIBUTE_SPLIT_CHAR).__len__() > 1:
                for attribute in attribute_value.split(ATTRIBUTE_SPLIT_CHAR):
                    knowledge_dict = raw_knowledge_dict.copy()
                    knowledge_dict.update({ATTRIBUTE_ZH: attribute})
                    knowledge_dict_ls.append(knowledge_dict)
            else:
                knowledge_dict_ls.append(raw_knowledge_dict)

            question_knowledge_ls = [QuestionKnowledge(question=standard_question,
                                                       knowledge_dict=knowledge_dict,
                                                       entity_type_zh_to_en_mapping=self._entity_type_zh_to_en_mapping)
                                     for knowledge_dict in knowledge_dict_ls
                                     ]
            # 输出 standard_question 对应的 知识列表（可能为多个）
            standard_question_to_knowledge_ls.update({standard_question: question_knowledge_ls})

        return standard_question_to_knowledge_ls

    # def _get_first_knowledge(self):
    #     first_knowledge_set = set()
    #     for standard_question,standard_question_knowledge_ls in self._standard_question_to_knowledge_ls.items():
    #         for standard_question_knowledge in standard_question_knowledge_ls:
    #             first_knowledge_set.add(standard_question_knowledge._first_knowledge)
    #     return list(first_knowledge_set)

    def _get_intents(self):
        intent_set = set()
        for standard_question, standard_question_knowledge_ls in self._standard_question_to_knowledge_ls.items():
            for standard_question_knowledge in standard_question_knowledge_ls:
                intent_set.add(standard_question_knowledge._intent)
        return list(intent_set)

    def _get_attributes(self):
        attribute_set = set()
        for standard_question, standard_question_knowledge_ls in self._standard_question_to_knowledge_ls.items():
            for standard_question_knowledge in standard_question_knowledge_ls:
                attribute_set.add(standard_question_knowledge._attribute)
        return list(attribute_set)

    def _get_knowledge_to_standard_questions(self, ) \
            -> Dict[Text, List[Text]]:
        knowledge_to_standard_questions = defaultdict(list)
        for standard_question, question_knowledge_ls in self._standard_question_to_knowledge_ls.items():
            # 判断每个标准问是否存在多个属性，对应多个知识
            for question_knowledge in question_knowledge_ls:
                knowledge_str = question_knowledge._knowledge_dict_to_string()
                standard_questions = knowledge_to_standard_questions[knowledge_str]
                standard_questions.append(standard_question)
                knowledge_to_standard_questions.update({knowledge_str: standard_questions})
        return knowledge_to_standard_questions

    def _get_knowledge_based_questions(self) -> Tuple[List[NamedTuple], List[NamedTuple]]:
        # 变更 列表类型为 namedtuple
        knowledge_based_standard_questions = []
        none_knowledge_based_standard_questions = []
        KnowledgeQuestionNamedtuple = namedtuple('knowledge_to_question', ['knowledge', 'standard_questions'])
        for knowledge_dict_str, standard_questions in self._knowledge_to_standard_questions.items():
            # 从 knowledge_dict_str 回复创建 QuestionKnowledge 对象
            knowledge = QuestionKnowledge.instance_from_knowledge_dict_str(knowledge_dict_str=knowledge_dict_str,
                                                                           entity_type_zh_to_en_mapping=self._entity_type_zh_to_en_mapping
                                                                           )
            if standard_questions.__len__() == 1:
                logger.debug(f"标准问可以使用knowledge唯一确定,标准问={standard_questions[0]}，knowledge={knowledge_dict_str}")
                # knowledge_based_standard_questions.append(standard_questions[0])
                knowledge_question_namedtuple = KnowledgeQuestionNamedtuple(knowledge=knowledge,
                                                                            standard_questions=standard_questions)
                knowledge_based_standard_questions.append(knowledge_question_namedtuple)
            else:
                logger.debug(f"标准问无法可以使用knowledge唯一确定,标准问={standard_questions}，knowledge={knowledge_dict_str}")

                knowledge_question_namedtuple = KnowledgeQuestionNamedtuple(knowledge=knowledge,
                                                                            standard_questions=standard_questions)
                none_knowledge_based_standard_questions.append(knowledge_question_namedtuple)
        logger.info(f"标准问可以使用knowledge唯一确定共{knowledge_based_standard_questions.__len__()},"
                    f"标准问无法可以使用knowledge唯一确定共{none_knowledge_based_standard_questions.__len__()},")
        return knowledge_based_standard_questions, none_knowledge_based_standard_questions

    def _get_knowledge_based_standard_questions_ratio(self):
        "标准问可以使用knowledge唯一确定的覆盖率为"
        # f"{knowledge_based_standard_questions.__len__() / self._standard_question_to_knowledge.__len__()}")
        return self._knowledge_to_standard_questions.__len__() / self._standard_questions.__len__()

    def get_actions_and_rules(self):
        self._actions, self._rules = self._build_actions_and_rules()
        return self._actions, self._rules

    def _build_actions_and_rules(self) -> Tuple[List[Action], List[Dict[Text, Any]]]:
        """
        根据 标注问 对应的 意图+属性+实体信息来建立规则，回复对应的 回答
        如果可以通过 意图+属性+实体信息 唯一确定的，我们直接使用对应的标注问作为 response。
        如果无法唯一确定，则返回为空字符

        bug： 当 意图 + 属性 正确，实体却和定义模板不一致的情况， text 返回为空
        fix： 增加一个正则的场景： 需要对每个意图增加一个正则表达，除非该意图已经唯一对应标注问了

        """
        # 对于可以唯一确定的标准问，建立规则
        action_name_to_action_dict = {}
        rules = []
        for knowledge_question_namedtuple in self._knowledge_based_standard_questions:
            knowledge = knowledge_question_namedtuple.knowledge
            standard_questions = knowledge_question_namedtuple.standard_questions
            for standard_question in standard_questions:
                action, rule = self._create_action_and_rule(standard_question, knowledge)
                if action:
                    logger.debug(f"使用知识可以唯一确定的标准问的规则,action:{action}")
                action_name_to_action_dict, rules = self._update_actions_and_rules(action, rule,
                                                                                   action_name_to_action_dict,
                                                                                   rules)

        # 对于不能确定的标注问？？
        for knowledge_question_namedtuple in self._none_knowledge_based_standard_questions:
            knowledge = knowledge_question_namedtuple.knowledge
            standard_questions = knowledge_question_namedtuple.standard_questions
            for standard_question in standard_questions:
                # 如何确立 rule ？ 返回 response 为空字符
                action, rule = self._create_action_and_rule(standard_question, question_knowledge=knowledge,
                                                            is_none_knowledge_based_standard_question=True)
                if action:
                    logger.info(f"使用知识无法唯一确定的标准问的规则,action:{action}")
                action_name_to_action_dict, rules = self._update_actions_and_rules(action, rule,
                                                                                   action_name_to_action_dict,
                                                                                   rules)

        for intent in self._intent_to_standard_questions.keys():
            action, rule = self._create_intent_corresponding_rule(intent=intent)
            if action:
                logger.debug(f"使用 intent 知识确定的规则,action:{action}")
            action_name_to_action_dict, rules = self._update_actions_and_rules(action, rule, action_name_to_action_dict,
                                                                               rules)

        actions = [action for action_name, action in action_name_to_action_dict.items()]

        return actions, rules

    def _create_action_and_rule(self, standard_question: Text,
                                question_knowledge: QuestionKnowledge,
                                is_none_knowledge_based_standard_question=False,
                                if_use_rule_condition=False,
                                wait_for_user_input=False,
                                use_custom_actions=False,
                                ) -> Tuple[Action, Dict[Text, Text]]:
        """
        使用 标准问 和 知识来建立 规则

        """
        # question_knowledge = self._standard_question_to_knowledge[standard_question]

        if is_none_knowledge_based_standard_question:
            # 标准问不能通过知识唯一确定的情况
            # response = DEFAULT_RESPONSE
            knowledge_as_string = question_knowledge._knowledge_as_string
            knowledge_to_standard_questions = self._knowledge_to_standard_questions[knowledge_as_string]
            try:
                assert knowledge_to_standard_questions.__len__() > 1
            except Exception as e:
                logger.error(f"标准问:{standard_question}不能通过知识唯一确定，因为这个知识对应的标准问存在多个的情况，但是现在对应的标准问只有一个")
            #  对于同一个知识的标注问拼接作为 response
            response = "/".join(knowledge_to_standard_questions)
            # 对于同一个知识作为 action_name
            action_name = question_knowledge._knowledge_as_string
        else:
            # 标准问可以通过知识唯一确定的情况
            # 使用 standard_question 作为 response
            response = standard_question
            # 使用 standard_question + knowledge : 作为 action_name :utter_{action_name}
            action_name = f"{standard_question}_{question_knowledge._knowledge_as_string}"

        # 使用 action_name 作为 rule name
        rule_name = action_name

        action = make_rasa_action(action_name=action_name, response=response)
        if question_knowledge._entity_num == 0:
            # 当使用  意图进行判断的action的时候
            rule = {"rule": rule_name,
                    "steps": [{'intent': question_knowledge._intent},
                              {'action': action.action_name}],
                    "wait_for_user_input": wait_for_user_input
                    }
        elif question_knowledge._entity_num == 1:
            # 判断是否存在 单个 实体信息
            # 当 entity信息使用判断的action的时候
            # https://github.com/rasahq/rasa/issues/8581
            rule = {"rule": rule_name,
                    # "condition":condition,
                    "steps": [{"intent": question_knowledge._intent},
                              # "entities": question_knowledge._entities}
                              {"slot_was_set": question_knowledge._entities},
                              {'action': action.action_name}],
                    "wait_for_user_input": wait_for_user_input
                    }
        elif question_knowledge._entity_num > 1:
            # TODO : 增加表单的情况（多个实体的情况）
            logger.warning(f"标注问={standard_question} 存在多个实体的情况，先不设置action:"
                           f"{question_knowledge._knowledge_as_string}")
            action = None
            rule = None
        return action, rule

    def _update_actions_and_rules(self, action, rule, action_name_to_action_dict, rules):
        if action and action.action_name not in action_name_to_action_dict:
            action_name_to_action_dict.update({action.action_name: action})
            rules.append(rule)
        return action_name_to_action_dict, rules

    def _create_intent_corresponding_rule(self, intent, wait_for_user_input=False):
        """
        bug： 当 意图 + 属性 正确，实体却和定义模板不一致的情况， text 返回为空
        fix： 增加一个正则的场景： 需要对每个意图增加一个正则表达，除非该意图已经唯一对应标注问了
        """
        # 讲 intent 作为知识 获取意图对应的标准问，判断意图是否对应一个标准问
        intent_as_knowledge = self._intent_as_knowledge(intent)

        # 对于同一个 intent_as_knowledge 作为 action_name
        action_name = intent_as_knowledge
        # 使用 action_name 作为 rule name
        rule_name = action_name

        standard_questions = self._knowledge_to_standard_questions.get(intent_as_knowledge)
        if standard_questions is None:
            # 说明没有 意图 作为知识直接对应的 标准问,需要增加该意图对应的规则，使得当 遇到该意图的时候，抛出其对应的所有标准问
            #  对于同一个知识的标注问拼接作为 response
            standard_questions = self._intent_to_standard_questions[intent]
            response = "/".join(standard_questions)
        elif standard_questions.__len__() == 1:
            # 说明 意图作为知识唯一的确定一个标准问，已经增加了规则
            return None, None
        else:
            response = "/".join(standard_questions)

        action = make_rasa_action(action_name=action_name, response=response)
        # 说明 意图作为 知识可以对应多个标准问，需要增加该意图对应的规则，使得当 遇到该意图的时候，抛出其对应的所有标准问
        rule = {"rule": rule_name,
                "steps": [{'intent': intent},
                          {'action': action.action_name}],
                "wait_for_user_input": wait_for_user_input
                }
        return action, rule

    # def _intent_as_knowledge(self, intent, intent_split_char='_', knowledge_combine_char="/"):
    #     """
    #     将 intent 组装成为 knowledge 的key
    #     TODO: 需要和 QuestionKnowledge._knowledge_dict_to_string() 方法保持一致的顺序
    #     """
    #     first_knowledge_and_attribute = intent.split(intent_split_char)
    #     if first_knowledge_and_attribute.__len__() == 1:
    #         first_knowledge = first_knowledge_and_attribute[0]
    #         intent_as_knowledge = f"{FIRST_KNOWLEDGE_ZH}-{first_knowledge}"
    #     elif first_knowledge_and_attribute.__len__() == 2:
    #         first_knowledge, attribute = first_knowledge_and_attribute
    #         intent_as_knowledge = f"{FIRST_KNOWLEDGE_ZH}-{first_knowledge}{knowledge_combine_char}{ATTRIBUTE_ZH}-{attribute}"
    #     else:
    #         raise ValueError(f"intent={intent}不符合定义规范")
    #     return intent_as_knowledge

    def _intent_as_knowledge(self, intent, intent_attribute_split_char=INTENT_ATTRIBUTE_SPLIT_CHAR,
                             knowledge_combine_char=KNOWLEDGE_COMBINE_CHAR):
        """
        将 intent 组装成为 knowledge 的key
        TODO: 需要和 QuestionKnowledge._knowledge_dict_to_string() 方法保持一致的顺序
        """
        intent_and_attribute = intent.split(intent_attribute_split_char)
        if intent_and_attribute.__len__() == 1:
            intent = intent_and_attribute[0]
            intent_as_knowledge = f"{INTENT_ZH}-{intent}"
        elif intent_and_attribute.__len__() == 2:
            intent, attribute = intent_and_attribute
            intent_as_knowledge = f"{INTENT_ZH}-{intent}{knowledge_combine_char}{ATTRIBUTE_ZH}-{attribute}"
        else:
            raise ValueError(f"intent={intent}不符合定义规范")
        return intent_as_knowledge

    def _get_intent_to_standard_questions(self) -> Dict[Text, List[Text]]:
        intent_to_standard_questions = defaultdict(set)
        for standard_question, question_knowledge_ls in self._standard_question_to_knowledge_ls.items():
            for question_knowledge in question_knowledge_ls:
                intent = question_knowledge._intent
                standard_questions = intent_to_standard_questions[intent]
                standard_questions.add(standard_question)
        intent_to_standard_questions_ls = {intent: list(standard_questions)
                                           for intent, standard_questions in intent_to_standard_questions.items()}
        return intent_to_standard_questions_ls

    def get_question_key_value(self, question: Text, key=Text):
        question_knowledge = self._standard_question_to_knowledge(question=question)
        return question_knowledge.get(key)


class IntentEntityInfoToResponse():
    """
    模型意图 对应的 entity 信息： entity 属性和 值，对应的 function_name（后面作为response）
    # { intent : {entity: entity_values},
        response:,
    }
    """

    def __init__(self, function_name_information_dict,
                 intent_column=INTENT,
                 # response_column= '二级知识点',
                 response_key=RESPONSE,  # SECONDARY_INTENT,
                 entity_key=ENTITY,
                 entity_values_key=ENTITY_VALUES
                 ):
        self.function_name_information_dict = function_name_information_dict
        self.intent_to_entity_mapping = {}

        self.intent_column = intent_column
        # self.response_column = response_column
        self.response_key = response_key
        self.entity_key = entity_key
        self.entity_values_key = entity_values_key

    def build(self) -> Dict[Text, Dict[Text, List[Text]]]:
        """
        从 function_name_information_dict   获取 intent 与 entity 的 对应关系
        """

        # 对于 function_name_information_dict 进行变换，获取 intent_entity_info
        intent_to_entity_mapping = {}
        for function_name, function_name_info in self.function_name_information_dict.items():
            intent = function_name_info.get(self.intent_column)
            # 获取 function_name 的实体信息
            entity = function_name_info.get(self.entity_key)
            entity_values = function_name_info.get(self.entity_values_key)
            # 获取 function_name 对应的response
            response = function_name_info.get(self.response_key)

            if intent is not None:
                self._add(intent_to_entity_mapping,
                          intent, entity, entity_values, response, )
        self.intent_to_entity_mapping = intent_to_entity_mapping
        return intent_to_entity_mapping

    def _add(self, intent_to_entity_mapping,
             intent, entity, entity_values, response):
        """
        {intent:
             {entity:
                  {entity_values_str_key:
                       {entity_values_key:entity_values,
                        response_key: response },
                   response_key: response
                   },
              response_key: response
              },
         }

        """

        pre_response = None
        if entity is None:
            # 当 intent 没有对应的 entity的时候，只需要 intent 就可以找到对应 response
            entity_info = intent_to_entity_mapping.get(intent, {})
            if entity_info is not None:
                # 获取之前的 response
                pre_response = entity_info.get(self.response_key)
            if pre_response is not None:
                # 之前的 response 和 现在的 response 应该相同
                if pre_response != response:
                    raise ValueError(f"intent={intent} 对应的 response={response},"
                                     f"但是pre_response={pre_response}已经存在")
                else:
                    logger.warning(f"intent={intent} 对应的已经存在 pre_response={pre_response},")
            else:
                entity_info.update({self.response_key: response})

                intent_to_entity_mapping.update(
                    {intent: entity_info}
                )
        elif entity_values is None:
            # 当 intent 存在对应的 entity的但是没有 entity_values,需要根据 intent + entity 属性来判断 response_intent
            # 意图对应的 entity 信息，默认为一个字典，key= entity属性： key = response_intent
            entity_info = intent_to_entity_mapping.get(intent, {})
            # 创建对应 entity
            pre_response = entity_info.get(entity, {}).get(self.response_key)
            if pre_response is not None:
                if pre_response != response:
                    raise ValueError(f"intent={intent}+ entity={entity} 对应的 response{response},"
                                     f"但是pre_response={pre_response} 已经存在")
                else:
                    logger.warning(f"intent={intent}+ entity={entity} 对应的已经存在相同的 response={pre_response}")
            else:
                entity_info.update(
                    {entity:
                         {self.response_key: response}
                     }
                )
                intent_to_entity_mapping.update(
                    {intent: entity_info})

        elif entity_values is not None:
            sorted_entity_values, entity_values_str_key = self._as_entity_values_str_key(entity_values)
            # 意图对应的 entity 信息，默认为一个字典，key= entity属性
            entity_info = intent_to_entity_mapping.get(intent, {})
            # entity_type 对应的 信息为 字典
            entity_type_info = entity_info.get(entity, {})
            # 查看 对应的  entity_values_str_key 是否已经存在 response
            pre_response = entity_type_info.get(entity_values_str_key, {}).get(self.response_key)
            if pre_response is not None:
                if pre_response != response:
                    raise ValueError(
                        f"intent={intent}+ entity={entity} + entity_values_str_key= {entity_values_str_key} "
                        f"对应的 response={response},"
                        f"但是pre_response={pre_response} 已经存在")
                else:
                    logger.warning(f"intent={intent}+ entity={entity} + entity_values_str_key= {entity_values_str_key}"
                                   f"对应已经存在相同的 response={pre_response}")

            else:
                entity_type_info.update(
                    {entity_values_str_key:
                         {self.entity_values_key: sorted_entity_values,
                          self.response_key: response}
                     }
                )
                entity_info.update(
                    {entity: entity_type_info}
                )

                intent_to_entity_mapping.update(
                    {intent: entity_info}
                )

    def _as_entity_values_str_key(self, entity_values: List[Text], sep='_'):
        assert isinstance(entity_values, list)
        sorted_entity_values = sorted(entity_values)
        entity_values_str_key = f"{sep}".join(sorted(entity_values))
        return sorted_entity_values, entity_values_str_key

    def __contains__(self, intent):
        # 重写 __contains__ 方法，用于判断 intent 是否在 该对象中
        if intent in self.intent_to_entity_mapping:
            return True
        else:
            return False

    def get_response(self, intent, entity=None, entity_values=None,
                     response_key='response'):
        """
        意图 + （slot 信息） 可以对应每个 response_intent,该 response_intent 返回给下级相似度模型
        """

        if intent not in self.intent_to_entity_mapping:
            raise ValueError(f"intent={intent}不在支持的意图范围内")
        elif entity is None:
            # if function_name_key  in self.intent_to_entity_mapping.get(intent):
            # 当没有 entity为空的时候，只需要更加 intent 就可以得到 response_intent
            response = self.intent_to_entity_mapping.get(intent).get(response_key)

        elif entity_values is None:
            # 当 entity_values 为空的时候，根据 intent + entity 来获取 function_name
            entity_info = self.intent_to_entity_mapping.get(intent).get(entity)
            response = entity_info.get(response_key)
        else:
            sorted_entity_values, entity_values_str_key = self._as_entity_values_key(entity_values)
            entity_values_info = self.intent_to_entity_mapping.get(intent).get(entity).get(entity_values_str_key)
            response = entity_values_info.get(response_key)
        assert response is not None
        return response

    def get_entity_info(self, intent: Text) -> List[Dict[Text, Text]]:
        """
        根据 intent 获取 entity 和 entity_values 和对应 response
        """
        intent_entity_mapping_ls = []
        if intent not in self.intent_to_entity_mapping:
            raise ValueError(f"intent={intent}不在支持的意图范围内")
        # 当 intent 没有对应的 entity的时候，只需要 intent 就可以找到对应 response
        intent_entity_info = self.intent_to_entity_mapping.get(intent)

        # 如果 intent 对应存在 response 增加这个信息到 intent_entity_mapping_ls
        if self.response_key in intent_entity_info:
            response = intent_entity_info.get(self.response_key)
            intent_entity_mapping_ls.append({self.intent_column: intent,
                                             self.entity_key: None,
                                             self.entity_values_key: None,
                                             self.response_key: response})

        entity_types = set(intent_entity_info.keys()).difference({self.response_key})
        # if entity_types.__len__() > 0:
        for entity_type in entity_types:
            entity_type_info = intent_entity_info.get(entity_type)
            entity_values_str_keys = set(entity_type_info.keys())
            if self.response_key in entity_values_str_keys:
                # 对于 intent + entity_type 对应的 response
                response = entity_type_info.get(self.response_key)
                intent_entity_mapping_ls.append({self.intent_column: intent,
                                                 self.entity_key: entity_type,
                                                 self.entity_values_key: None,
                                                 self.response_key: response})

            entity_values_str_keys = entity_values_str_keys.difference({self.response_key})
            for entity_values_str_key in entity_values_str_keys:
                # 对于 intent + entity_type + entity_values 来确定的 response
                entity_values_dict = entity_type_info.get(entity_values_str_key)

                response = entity_values_dict.get(self.response_key)
                entity_values = entity_values_dict.get(self.entity_values_key)
                intent_entity_mapping_ls.append({self.intent_column: intent,
                                                 self.entity_key: entity_type,
                                                 self.entity_values_key: entity_values,
                                                 self.response_key: response})

        return intent_entity_mapping_ls


# def get_intent_from_labeled_data(row):
#     """
#     从标注数据中获取 模型使用的 意图
#     """
#     first_knowledge = row[FIRST_KNOWLEDGE]
#
#     attribute = row[ATTRIBUTE]
#     try:
#         if not isinstance(first_knowledge, str):
#             return
#         elif (not isinstance(attribute, str) and pd.isna(attribute)) or (attribute is None):
#             # first_knowledge 为其他或者转人工的情况，不区分属性
#             intent = f"{first_knowledge}"
#         elif isinstance(first_knowledge, str) and isinstance(attribute, str):
#             intent = f"{first_knowledge.strip()}_{attribute.strip()}"
#         else:
#             raise ValueError(f"first_knowledge={first_knowledge},attribute={attribute}值有错误！")
#         return intent
#     except Exception as e:
#         logger.error(f"意图设置错误:{e}")


def get_intent_from_labeled_data(row):
    """
    从标注数据中获取 模型使用的 意图
    """
    intent = row[INTENT]

    attribute = row[ATTRIBUTE]
    try:
        if not isinstance(intent, str):
            return
        elif (not isinstance(attribute, str) and pd.isna(attribute)) or (attribute is None):
            # first_knowledge 为其他或者转人工的情况，不区分属性
            intent = f"{intent}"
        elif isinstance(intent, str) and isinstance(attribute, str):
            intent = f"{intent.strip()}_{attribute.strip()}"
        else:
            raise ValueError(f"intent={intent},attribute={attribute}值有错误！")
        return intent
    except Exception as e:
        logger.error(f"意图设置错误:{e}")


def match_entity_type_columns(columns):
    return [column for column in columns if re.match(pattern=f"{ENTITY_TYPE}_(\d)", string=column)]


if __name__ == '__main__':
    task = 'kbqa_hsnlp_faq_bot'
    dataset = 'kbqa_hsnlp_faq_data'
    financial_term_filename = 'financial_entity.yml'
    raw_data_filename = 'standard_to_extend_question_data-20211130.xlsx'
    faq_knowledge_filename = 'hsnlp-基于知识图谱的意图和实体定义体系-20211221.xlsx'
    sheet_name = 'haitong-ivr标准问与意图属性实体对应关系-1221'
    from conf.config_parser import CORPUS

    from utils.common import init_logger

    logger = init_logger(output_dir=f'output/{task}')
    financial_term_data_path = os.path.join(CORPUS, dataset, financial_term_filename)
    standard_question_knowledge_file_path = os.path.join(CORPUS, dataset, faq_knowledge_filename)
    # 读取实体信息
    # entity_data = EntityData(data_path=financial_term_data_path)
    # 读取海通的标注定义信息
    haitong_ivr_standard_question_knowledge = StandardQuestionKnowledgeData(
        standard_question_knowledge_file_path=standard_question_knowledge_file_path,
        sheet_name=sheet_name,
    )
