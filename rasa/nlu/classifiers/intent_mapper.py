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
from pathlib import Path
from typing import Optional, Any, Text, Dict, List, Set
import rasa
from data_process.dataset.hsnlp_faq_knowledge_dataset import  StandardQuestionKnowledgeData # FIRST_KNOWLEDGE_ZH
from rasa.shared.nlu.constants import ENTITIES, INTENT, OUTPUT_INTENT, \
    INTENT_TO_OUTPUT_MAPPING_FILENAME, INTENT_RANKING_KEY, INTENT_NAME_KEY, PREDICTED_CONFIDENCE_KEY
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.components import Component
import rasa.utils.io as io_utils
from rasa.shared.utils.io import create_directory_for_file
import logging

logger = logging.getLogger(__name__)


class IntentToOutputMapper(Component):
    """
    将模型的意图变换为 大意图
    """

    # defaults = {
    #     "intent_to_output_mapping_filename":'intent_to_output_mapping'
    # }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None,
                 intent_to_output_mapping: Dict[Text, Text] = None):
        super(IntentToOutputMapper, self).__init__(component_config)
        self.intent_to_output_mapping_dir = component_config.get('intent_to_output_mapping_dir')
        self.function_name_information_filename = component_config.get('function_name_information_filename')
        self.intent_to_output_mapping_filename = component_config.get('intent_to_output_mapping_filename',
                                                                      INTENT_TO_OUTPUT_MAPPING_FILENAME)
        self.intent_to_output_mapping = intent_to_output_mapping or {}

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        """
        从原始数据中获取 intent_to_output_mapping
        """
        if self.intent_to_output_mapping_dir is None or self.function_name_information_filename is None:
            raise ValueError(f"intent_to_output_mapping_dir 或者 function_name_information_filename 为空")
        else:
            function_name_information_file = Path(
                self.intent_to_output_mapping_dir) / self.function_name_information_filename
            if not os.path.exists(function_name_information_file):
                raise FileNotFoundError(f"{function_name_information_file}不存在")
            else:
                try:
                    # function_name_information = rasa.shared.utils.io.read_json_file(function_name_information_file)
                    # intent_to_output_mapping = {function_name_info[INTENT]: function_name_info[FIRST_KNOWLEDGE_ZH]
                    #                             for function_name, function_name_info in
                    #                             function_name_information.items()
                    #                             }

                    standard_question_knowledge_data = StandardQuestionKnowledgeData(
                        standard_question_knowledge_file_path=str(function_name_information_file))
                    intent_to_output_mapping = {}
                    # 修改 _standard_question_to_knowledge to _standard_question_to_knowledge_ls
                    for standard_question, question_knowledge_ls in standard_question_knowledge_data._standard_question_to_knowledge_ls.items():
                        for question_knowledge in  question_knowledge_ls:
                            intent = question_knowledge._intent
                            output_intent = question_knowledge._first_knowledge
                            intent_to_output_mapping.update({intent:output_intent})

                    intents = set(intent_to_output_mapping.keys())
                    trained_intents = set(training_data.intents)
                    # 标注的数据可能意图
                    # assert trained_intents.issubset(intents)
                    difference_intents = trained_intents.difference(intents)
                    if difference_intents.__len__()>0:
                        logger.warning(f"训练数据中的意图于标准问定义系统中的差别={difference_intents}")
                    self.intent_to_output_mapping = intent_to_output_mapping
                except Exception as e:
                    raise IOError(f"读取{function_name_information_file}发生错误:{e}")

    def process(self, message: Message, **kwargs: Any) -> None:
        intent_info = message.get(INTENT)
        intent_name = intent_info.get(INTENT_NAME_KEY)
        if intent_name in self.intent_to_output_mapping:
            # 使用  intent_to_output_mapping 输出对应的大意图
            output_intent_name = self.intent_to_output_mapping.get(intent_name)
            # 计算 output_intent probability: 将 intent_ranking 中排序的intent_name 对应相同的输出意图的 confidence 相加
            same_output_confidences = [intent_info[PREDICTED_CONFIDENCE_KEY]
                                       for intent_info in message.get(INTENT_RANKING_KEY)
                                       if self.intent_to_output_mapping.get(
                    intent_info[INTENT_NAME_KEY]) == output_intent_name]
            output_confidences = sum(same_output_confidences)
            output_intent_info = {INTENT_NAME_KEY: output_intent_name, PREDICTED_CONFIDENCE_KEY: output_confidences}
        else:
            # 如果没有找到对应的意图，就返回原来的意图作为大意图
            output_intent_info = intent_info
        message.set(OUTPUT_INTENT, output_intent_info, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """
        保存 intent_to_output_mapping 到模型文件目录中
        """
        intent_to_output_mapping_file = Path(model_dir) / f"{file_name}.{self.intent_to_output_mapping_filename}.json"
        rasa.shared.utils.io.dump_obj_as_json_to_file(intent_to_output_mapping_file, self.intent_to_output_mapping)
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
        intent_to_output_mapping_filename = meta.get("intent_to_output_mapping_filename")
        intent_to_output_mapping_file = Path(model_dir) / f"{file_name}.{intent_to_output_mapping_filename}.json"
        if os.path.exists(intent_to_output_mapping_file):
            intent_to_output_mapping = rasa.shared.utils.io.read_json_file(intent_to_output_mapping_file)
            return IntentToOutputMapper(meta, intent_to_output_mapping)
        else:
            return IntentToOutputMapper(meta)
