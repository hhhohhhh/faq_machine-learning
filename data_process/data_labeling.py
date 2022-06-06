#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/23 14:46

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/23 14:46   wangfc      1.0         None
"""
import logging
from typing import Text, List, Tuple
from collections import OrderedDict
import tqdm
import pandas as pd

from data_process.dataset.hsnlp_faq_knowledge_dataset import EntityData, ENTITY_TYPE, ENTITY_VALUE, ENTITY_START, \
    ENTITY_END
from models.trie_tree import TrieTree, Entity

logger = logging.getLogger(__name__)


# entity_attribute_to_value_dict = {"block_kz": ["科创版", "科创板"],
#                                   "block_cy": ['创业板', '创业版'],
#                                   "stock": ['海通证券'],
#                                   "zsstock": ["恒生指数"]}


class EntityLabeler():
    def __init__(self, entity_data: EntityData):
        self.entity_data = entity_data

        self.entity_start = ENTITY_START
        self.entity_end = ENTITY_END
        self.entity_type = ENTITY_TYPE
        self.entity_value = ENTITY_VALUE

        self.entity_type_pattern = f"{self.entity_type}_(\d)"
        self.entity_value_pattern = f"{self.entity_value}_(\d)"

    def apply_entity_labeling(self, data: pd.DataFrame,
                              text_column: Text,
                              rasa_labeled_text_format: bool = True,
                              add_entity_extracted_columns: bool = False,
                              description='实体标注'):
        """
        增加 entity 的标注：

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
        entity_extracted_ls = []
        for index in tqdm.tqdm(range(data.__len__()), desc=description):
            text = data.iloc[index].loc[text_column]
            # 使用  tire_tree 进行 实体标注:
            # TODO: 增加输出实体的信息为一个字典： key: entity_start_01,entity_end_01, entity_type_01,entity_value_01
            new_text, entities = entity_labeling_with_tire_tree(text,
                                                                self.entity_data._entity_tire_tree,
                                                                rasa_labeled_text_format=rasa_labeled_text_format)
            new_texts.append(new_text)
            entity_extracted_ls.append(entities)
        # 将实体标注的数据
        if rasa_labeled_text_format:
            data.loc[:, text_column] = new_texts

        if add_entity_extracted_columns:
            # 增加实体的列：
            entity_ls = self._transform_entity_extracted_ls(entity_extracted_ls)
            # 转换为 dataframe
            entity_df = pd.DataFrame(entity_ls)
            # 合并
            data = pd.concat([data, entity_df], axis=1)

        return data

    def _transform_entity_extracted_ls(self, entity_extracted_ls: List[List[Entity]])-> List[OrderedDict]:
        """
        将 List[Entity] 转换为需要的字典
        """
        entity_ls = []
        for entities in entity_extracted_ls:
            entities_dict = OrderedDict()
            for entity_index, entity in enumerate(entities):
                start_position = entity.start_position
                end_position = entity.end_position
                entity_type = entity.entity_type
                entity_value = entity.text
                entity_dict = {f"{key}_{entity_index}": value
                               for key, value in
                               zip([self.entity_start, self.entity_end, self.entity_type, self.entity_value],
                                   [start_position, end_position, entity_type, entity_value])
                               }
                entities_dict.update(entity_dict)
            entity_ls.append(entities_dict)
        return entity_ls

def entity_labeling(text, entity_attribute_to_value_dict):
    for entity_attribute, entity_values in entity_attribute_to_value_dict.items():
        for entity_value in entity_values:
            if entity_value in text:
                # start_position = text.find(entity_value)
                entity_value_labeled = f"[{entity_value}]({entity_attribute})"
                text = text.replace(entity_value, entity_value_labeled)
                logger.info(f"text={text},entity_value={entity_value},entity_value_labeled={entity_value_labeled}")
    return text


def entity_labeling_with_tire_tree(text: Text, entity_tire_tree: TrieTree,
                                   rasa_labeled_text_format=True) -> Tuple[Text, List[Entity]]:
    """
    @time:  2021/12/24 10:46
    @author:wangfc
    @version:
    @description: 使用 entity tire tree 对 句子进行 entity 的标注
    rasa_labeled_text_format: 用于表示输出为rasa 的标注格式

    @params:
    @return:
    增加 List[Entity]的输出
    """

    entity_ls = entity_tire_tree.entity_labeling(text)
    if entity_ls.__len__() == 0:
        sequence = text
    elif rasa_labeled_text_format:
        # 将 sequence 标注为 rasa_labeled_text_format
        sequence = ''
        offset = 0
        entity_index = 0
        while offset < text.__len__():
            if entity_index < entity_ls.__len__():
                entity = entity_ls[entity_index]
                if offset < entity.start_position:
                    ch = text[offset]
                    sequence += ch
                    offset += 1
                else:
                    entity_str = text[entity.start_position: entity.end_position]
                    assert entity_str == entity.text
                    entity_with_label = f"[{entity_str}]({entity.entity_type})"
                    sequence += entity_with_label
                    # 更新 搜索的 offset
                    offset = entity.end_position
                    # 更新匹配的 entity_index
                    entity_index += 1
            else:
                # 当遍历完所有的 entity的时候
                sequence += text[offset:]
                offset = text.__len__()
    else:
        sequence = text

    return sequence, entity_ls

# if __name__ == '__main__':
#     text = "你好，我想开通创业版"
#     entity_labeling(text,entity_attribute_to_value_dict)
