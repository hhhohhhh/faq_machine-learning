#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/19 17:57 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 17:57   wangfc      1.0         None
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Text

from data_process.training_data.message import Message
from data_process.training_data.training_data import TrainingData
from models import pattern_utils
from models.extractors.extrator import EntityExtractor
from models.model import Metadata
from models.nlu.nlu_config import NLUModelConfig
from models.nlu.nlu_contants import ENTITIES, TEXT, ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_START, ENTITY_ATTRIBUTE_END, \
    ENTITY_ATTRIBUTE_VALUE
from utils.io import raise_warning, read_json_file, dump_obj_as_json_to_file

logger = logging.getLogger(__name__)


class RegexEntityExtractor(EntityExtractor):
    """Searches for entities in the user's message using the lookup tables and regexes
    defined in the training data."""

    defaults = {
        # text will be processed with case insensitive as default
        "case_sensitive": False,
        # use lookup tables to extract entities
        "use_lookup_tables": True,
        # use regexes to extract entities
        "use_regexes": True,
        # use match word boundaries for lookup table
        "use_word_boundaries": True,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        patterns: Optional[List[Dict[Text, Text]]] = None,
    ):
        """Extracts entities using the lookup tables and/or regexes defined."""
        super(RegexEntityExtractor, self).__init__(component_config)

        self.case_sensitive = self.component_config["case_sensitive"]
        self.patterns = patterns or []

    def train(
        self,
        training_data: TrainingData,
        config: Optional[NLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.patterns = pattern_utils.extract_patterns(
            training_data,
            use_lookup_tables=self.component_config["use_lookup_tables"],
            use_regexes=self.component_config["use_regexes"],
            use_only_entities=True,
            use_word_boundaries=self.component_config["use_word_boundaries"],
        )

        if not self.patterns:
            raise_warning(
                "No lookup tables or regexes defined in the training data that have "
                "a name equal to any entity in the training data. In order for this "
                "component to work you need to define valid lookup tables or regexes "
                "in the training data."
            )

    def process(self, message: Message, **kwargs: Any) -> None:
        if not self.patterns:
            return

        extracted_entities = self._extract_entities(message)
        extracted_entities = self.add_extractor_name(extracted_entities)

        message.set(
            ENTITIES, message.get(ENTITIES, []) + extracted_entities, add_to_output=True
        )


    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message."""
        entities = []

        flags = 0  # default flag
        if not self.case_sensitive:
            flags = re.IGNORECASE

        for pattern in self.patterns:
            matches = re.finditer(pattern["pattern"], message.get(TEXT), flags=flags)
            matches = list(matches)

            for match in matches:
                start_index = match.start()
                end_index = match.end()
                entities.append(
                    {
                        ENTITY_ATTRIBUTE_TYPE: pattern["name"],
                        ENTITY_ATTRIBUTE_START: start_index,
                        ENTITY_ATTRIBUTE_END: end_index,
                        ENTITY_ATTRIBUTE_VALUE: message.get(TEXT)[
                            start_index:end_index
                        ],
                    }
                )

        return entities

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["RegexEntityExtractor"] = None,
        **kwargs: Any,
    ) -> "RegexEntityExtractor":
        """Loads trained component (see parent class for full docstring)."""
        file_name = meta.get("file")
        regex_file = os.path.join(model_dir, file_name)

        if os.path.exists(regex_file):
            patterns = read_json_file(regex_file)
            return RegexEntityExtractor(meta, patterns=patterns)

        return RegexEntityExtractor(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        file_name = f"{file_name}.json"
        regex_file = os.path.join(model_dir, file_name)
        dump_obj_as_json_to_file(regex_file, self.patterns)

        return {"file": file_name}
