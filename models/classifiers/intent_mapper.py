#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/5 11:32 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/5 11:32   wangfc      1.0         None
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, Any, Text, Dict, List, Set
from data_process.dataset.hsnlp_faq_knowledge_dataset import StandardQuestionKnowledgeData, \
    ATTRIBUTE  # FIRST_KNOWLEDGE_ZH
# from rasa.shared.nlu.constants import ENTITIES, INTENT, OUTPUT_INTENT, \
#     INTENT_TO_OUTPUT_MAPPING_FILENAME, INTENT_RANKING_KEY, INTENT_NAME_KEY, PREDICTED_CONFIDENCE_KEY
# from rasa.nlu.config import RasaNLUModelConfig
# from rasa.shared.nlu.training_data.message import Message
# from rasa.shared.nlu.training_data.training_data import TrainingData
# from rasa.nlu.components import Component
# import rasa.utils.io as io_utils
# from rasa.shared.utils.io import create_directory_for_file
import logging

from data_process.training_data.message import Message
from data_process.training_data.training_data import TrainingData
from models.components import Component
from models.nlu.nlu_config import RasaNLUModelConfig


# from rasa.shared.nlu.constants import INTENT_TO_OUTPUT_MAPPING_FILENAME
from rasa.shared.nlu.constants import OUTPUT_INTENT
from utils.constants import INTENT_TO_OUTPUT_MAPPING_FILENAME, INTENT, INTENT_NAME_KEY, PREDICTED_CONFIDENCE_KEY, \
    INTENT_RANKING_KEY, INTENT_MAPPING_KEY, ATTRIBUTE_MAPPING_KEY, INTENT_ATTRIBUTE_MAPPING_FILENAME
from utils.io import dump_obj_as_json_to_file, read_json_file

logger = logging.getLogger(__name__)


class IntentAttributeMapper(Component):
    """
    将模型的意图变换为 大意图
    """
    defaults = {
        "intent_attribute_mapping_dir": None,
        "intent_attribute_mapping_filename":"intent_attribute_mapping.json"
    }
    def __init__(self, component_config: Optional[Dict[Text, Any]] = None,
                 intent_attribute_mapping: Dict[Text, Text] = None):
        super(IntentAttributeMapper, self).__init__(component_config)
        self.intent_attribute_mapping_dir = component_config.get('intent_attribute_mapping_dir')
        self.intent_attribute_mapping_filename = component_config.get('intent_attribute_mapping_filename',
                                                                      INTENT_ATTRIBUTE_MAPPING_FILENAME)
        self.intent_attribute_mapping = intent_attribute_mapping or {}

        self.intent_to_attribute_dict , self.intent_mapping , self.attribute_mapping = \
            self._parse_intent_attribute_mapping(self.intent_attribute_mapping)


    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        """
        从原始数据中获取 intent_to_output_mapping
        """
        if self.intent_attribute_mapping_dir is None or self.intent_attribute_mapping_filename is None:
            raise ValueError(f"intent_attribute_mapping_dir 或者 intent_to_output_mapping_filename 为空")
        else:
            intent_attribute_mapping_file = Path(
                self.intent_attribute_mapping_dir) / self.intent_attribute_mapping_filename
            if not os.path.exists(intent_attribute_mapping_file):
                raise FileNotFoundError(f"{intent_attribute_mapping_file}不存在")
            else:
                try:
                    self.intent_attribute_mapping = read_json_file(intent_attribute_mapping_file)
                    intent_to_attribute_dict, intent_mapping,attribute_mapping =  self._parse_intent_attribute_mapping(self.intent_attribute_mapping)

                    intents = set(intent_mapping.keys())
                    trained_intents = set(training_data.intents)

                    attributes = set(attribute_mapping.keys())
                    trained_attributes = set(attribute_mapping.keys())
                    # 标注的数据可能意图
                    assert intents.issubset(trained_intents)
                    assert attributes.issubset(trained_attributes)
                    self.intent_mapping = intent_mapping
                    self.attribute_mapping = attribute_mapping
                except IOError as e:
                    raise IOError(f"读取{intent_attribute_mapping_file}发生错误:{e}")
                except AssertionError as e:
                    raise IOError(f"映射的意图或者属性不是训练数据的子集，无法进行映射:{e}")
                except Exception as e:
                    raise (f"解析意图和属性的映射文件{intent_attribute_mapping_file}时发生错误：{e}")

    @classmethod
    def _parse_intent_attribute_mapping(cls,intent_attribute_mapping:Dict)-> [Dict[Text,List[Text]],
                                                                            Dict[Text,Text],Dict[Text,Text]]:
        """
        解析 intent_attribute_mapping
        """
        intent_to_attribute_dict, intent_mapping, attribute_mapping =None,None,None

        if intent_attribute_mapping:
            intent_mapping = {source_intent:target_intent for target_intent,source_intent
                              in  intent_attribute_mapping.get(INTENT_MAPPING_KEY).items()}
            attribute_mapping_info = intent_attribute_mapping.get(ATTRIBUTE_MAPPING_KEY)
            # 解析数据
            intent_to_attribute_dict = defaultdict(list)
            attribute_mapping = {}
            for intent,attribute_mapping_dict in attribute_mapping_info.items():
                corresponding_attributes = intent_to_attribute_dict[intent]
                for target_attribute,source_attribute_ls in attribute_mapping_dict.items():
                    for source_attribute in source_attribute_ls:
                        attribute_mapping.update({source_attribute:target_attribute})
                    corresponding_attributes.append(target_attribute)
        return intent_to_attribute_dict, intent_mapping,attribute_mapping



    def process(self, message: Message, **kwargs: Any) -> None:
        intent_info = message.get(INTENT)
        intent_name = intent_info.get(INTENT_NAME_KEY)

        if intent_name in self.intent_mapping:
            # 使用  intent_mapping 输出对应的意图
            mapped_intent_name = self.intent_mapping.get(intent_name)
            intent_info.update({INTENT_NAME_KEY: mapped_intent_name})
            intent_info = self._add_component_name(intent_info,component_type='mapper')
            message.set(INTENT, intent_info, add_to_output=True)

        attribute_info = message.get(ATTRIBUTE)
        attribute_name = attribute_info.get(INTENT_NAME_KEY)
        if attribute_name in self.attribute_mapping:
            # 使用  attribute_mapping 输出对应的属性
            mapped_attribute_name = self.attribute_mapping.get(attribute_name)
            attribute_info.update({INTENT_NAME_KEY: mapped_attribute_name})
            attribute_info = self._add_component_name(attribute_info, component_type='mapper')
            message.set(ATTRIBUTE, attribute_info, add_to_output=True)


    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """
        保存 intent_to_output_mapping 到模型文件目录中
        """
        intent_attribute_mapping_file = Path(model_dir) / f"{file_name}.{self.intent_attribute_mapping_filename}"
        dump_obj_as_json_to_file(intent_attribute_mapping_file, self.intent_attribute_mapping)
        return {'file': file_name}

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Text,
            model_metadata: Optional["Metadata"] = None,
            cached_component: Optional["Component"] = None,
            **kwargs: Any,
    ) -> "Component":
        """
        从模型文件 加载 intent_to_output_mapping
        """
        file_name = meta.get('file')
        intent_attribute_mapping_filename = meta.get("intent_attribute_mapping_filename")
        intent_attribute_mapping_file = Path(model_dir) / f"{file_name}.{intent_attribute_mapping_filename}"
        if os.path.exists(intent_attribute_mapping_file):
            intent_attribute_mapping= read_json_file(intent_attribute_mapping_file)
            return IntentAttributeMapper(meta, intent_attribute_mapping)
        else:
            return IntentAttributeMapper(meta)
