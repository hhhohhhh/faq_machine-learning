#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/4 17:38 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/4 17:38   wangfc      1.0         None

Components go through the three main stages:
create: initialization of the component before the training
train: the component trains itself using the context and potentially the output of the previous components
persist: saving the trained component on disk for the future use

the implementation of the main methods of the class:
init: initialization of the component
train: a method which is responsible for training the component
process: a method which will parse incoming user messages
persist: a method which will save a trained component on disk for later use


context
    Before the first component is initialized, a so-called context  is created which is used to pass the information between the components.

defining the main details:
name: the name of the component
provides: what output the custom component produces
requires: what attributes of the message this component requires
defaults: default configuration parameters of a component
language_list: a list of languages compatible with the component


"""
import os
from pathlib import Path
from typing import Optional, Any, Text, Dict, List, Set
import rasa
from rasa.shared.nlu.constants import ENTITIES, ENTITY_ATTRIBUTE_TYPE, EXTRACTOR
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.components import Component
import rasa.utils.io as io_utils
from rasa.shared.utils.io import create_directory_for_file
import logging

logger = logging.getLogger(__name__)


class EntitySelector(Component):
    defaults = {
        # 默认需要保留的抽取器
        "keep_extractor": 'RegexEntityExtractor',
        'regex_entities_filename': 'regex_entities'
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None,
                 regex_entities: List[Text] = None):
        super(EntitySelector, self).__init__(component_config)
        self.keep_extractor = component_config.get('keep_extractor')
        self.regex_entities_filename = component_config.get('regex_entities_filename', 'regex_entities')
        self.regex_entities = regex_entities or []

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        # 获取训练的时候 获取 regex_entities
        self.regex_entities = self._get_regex_entities(training_data)

    def process(self, message: Message, **kwargs: Any) -> None:
        """
        当多个 entity extrator 出现抽取同一个 entity attribute的时候，比如 block，
        如果 block 出现在 regex express 中，我们默认选择其抽取的结果，否认使用其他模型抽取的结果
        {'entity': 'block',
         'start': 3,
         'end': 9,
         'value': '创业板',
         'extractor': 'RegexEntityExtractor',
         'processors': ['EntitySynonymMapper']}

        """
        # entity = ['entity': 'block','extractor': ]
        entities = message.get(ENTITIES)
        # 查询相同的 entities 存在 多个抽取的情况，并且抽取器 不是正则的情况
        duplicate_entity_attributes_set = self._get_duplicate_entity_attributes_set(entities)

        if self.keep_extractor and self.regex_entities and duplicate_entity_attributes_set:
            # 对于存在 regex_entities 的情况
            updated_entities = []
            for entity in entities:
                # 如果 entity 属于 duplicate_entity_attributes_set 的时候，排除其他的
                if entity.get(ENTITY_ATTRIBUTE_TYPE) not in duplicate_entity_attributes_set \
                        or entity.get(EXTRACTOR).lower() == self.keep_extractor.lower():
                    updated_entities.append(entity)
            message.set(ENTITIES, updated_entities)

    def _get_duplicate_entity_attributes_set(self, entities) -> Set[Text]:
        duplicate_entity_attributes_set = set()
        entity_attributes_set = set()
        entity_attributes = [entity.get(ENTITY_ATTRIBUTE_TYPE) for entity in entities]
        for entity_attribute in entity_attributes:
            if entity_attribute not in entity_attributes_set:
                entity_attributes_set.add(entity_attribute)
            else:
                duplicate_entity_attributes_set.add(entity_attribute)
        return duplicate_entity_attributes_set

    def _get_regex_entities(self, training_data: TrainingData) -> List[Text]:

        """
        # 过滤 regex 中的 entity
        """
        from rasa.nlu.utils.pattern_utils import extract_patterns
        patterns = extract_patterns(training_data)
        pattern_names = set([pattern.get('name') for pattern in patterns])
        entities = training_data.entities
        regex_entities = [entity for entity in entities if entity in pattern_names]
        return regex_entities

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model （这个只是一个 regex_entities 列表 ）into the passed directory."""
        model_dir = Path(model_dir)
        regex_entities_file = model_dir / f"{file_name}.{self.regex_entities_filename}.json"
        rasa.shared.utils.io.dump_obj_as_json_to_file(regex_entities_file, self.regex_entities)

        return {"file": file_name}

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Text,
            model_metadata: Optional["Metadata"] = None,
            cached_component: Optional["Component"] = None,
            **kwargs: Any,
    ) -> "Component":
        """Loads the trained model from the provided directory."""
        if not meta.get("file"):
            logger.debug(
                f"Failed to load model for '{cls.__name__}'. "
                f"Maybe you did not provide enough training data and no model was "
                f"trained or the path '{os.path.abspath(model_dir)}' doesn't exist?"
            )
            return cls(component_config=meta)
        file_name = meta.get("file")
        regex_entities_filename = meta.get('regex_entities_filename')
        regex_entities_file = Path(model_dir) / f"{file_name}.{regex_entities_filename}.json"
        if os.path.exists(regex_entities_file):
            regex_entities = rasa.shared.utils.io.read_json_file(regex_entities_file)
            return EntitySelector(meta, regex_entities=regex_entities)
        else:
            return EntitySelector(meta)
