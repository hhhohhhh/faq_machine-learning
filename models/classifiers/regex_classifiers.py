#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/19 9:19 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 9:19   wangfc      1.0         None
"""
import os
import re
from typing import Optional, Dict, Text, Any, List, Union, NamedTuple,TYPE_CHECKING
import time
from collections import namedtuple
import hashlib

from apps.intent_attribute_classifier_apps.intent_attribute_classifier_app_constants import PREDICT_INTENT, \
    PREDICT_INTENT_CONFIDENCE, \
    PREDICT_ATTRIBUTE, PREDICT_ATTRIBUTE_CONFIDENCE
from data_process.dataset.hsnlp_faq_knowledge_dataset import ATTRIBUTE
from data_process.training_data.message import Message
from data_process.training_data.training_data import TrainingData
if TYPE_CHECKING:
    from models.model import Metadata
from models.nlu.nlu_config import NLUModelConfig
from utils.constants import INTENT, TEXT, INTENT_AND_ATTRIBUTE, RAW_TEXT, ID

from utils.io import raise_warning, read_json_file, dump_obj_as_json_to_file, is_likely_json_file, read_yaml_file, \
    is_likely_yaml_file
from models.classifiers.classifier import IntentClassifier
from utils.exceptions import InvalidRuleException
import logging

from utils.time import get_current_time

logger = logging.getLogger(__name__)

INDEX  = "index"
PATTERN = "pattern"
CREATE_TIME = 'create_time'

Pattern = namedtuple("pattern_data",[INDEX,ID,PATTERN,INTENT,ATTRIBUTE,CREATE_TIME])



class PatternData():
    def __init__(self,data_dir,patterns:Optional[List[Pattern]]=[]):
        self.data_dir = data_dir
        self.patterns = patterns or self._load()

    def _load(self) -> List[Pattern]:
        from utils.io import get_files_from_dir
        pathes = get_files_from_dir(self.data_dir)
        all_patterns = []
        for path in pathes:
            patterns = self._load_patterns_from_file(path=path)
            if patterns:
                all_patterns.extend(patterns)
        return all_patterns


    @staticmethod
    def _load_patterns_from_file(path, regex_key='regex') -> List[Pattern]:
        """
        @time:  2022/1/20 10:22
        @author:wangfc
        @version:
        @description: 从文件中直接读取

        @params:
        @return:
        """
        if os.path.exists(path):
            try:
                if is_likely_json_file(path):
                    patterns = read_json_file(path)
                elif is_likely_yaml_file(path):
                    patterns = read_yaml_file(path).get(regex_key)
                # dict 转换  Pattern 对象
                patterns = [add_pattern_info(pattern,index=index) for index,pattern in enumerate(patterns)]
                patterns = [Pattern(**pattern) for pattern in patterns]

                return patterns
            except Exception as e:
                logger.error(f"意图和属性的正则表达式{path}读取发生错误,请检查文件格式：{e}")
        else:
            logger.error(f"意图和属性的正则表达式未找到{path}")

    def __len__(self):
        return self.patterns.__len__()



    def update_patterns(self,new_pattens_ls:List[Dict[Text,Text]]):
        """
        @time:  2022/3/14 14:11
        @author:wangfc
        @version:
        @description: 更新的 pattern 优先于 原来的 pattern

        @params:
        @return:
        """
        new_patterns = [Pattern(**pattern) for pattern in new_pattens_ls]
        patterns = new_patterns +self.patterns
        # patterns = sorted(new_patterns + self.patterns,key=lambda x: x.index,reverse=True)
        self.patterns = patterns



def add_pattern_info(pattern: Dict[Text, Text],index:int):
    """给更新的数据增加信息:
    pattern_id, time"""
    if INDEX not in pattern:
        pattern.update({INDEX: index})
    if ID not in pattern:
        pattern_str = pattern.get(PATTERN)
        md5 = hashlib.md5()
        md5.update(pattern_str.encode('utf-8'))
        id = md5.hexdigest()
        pattern.update({"id": id})
    if CREATE_TIME not in pattern:
        current_time = get_current_time()
        pattern.update({"create_time": current_time})
    return pattern


class IntentAttributeRegexClassifier(IntentClassifier):
    """
    @time:  2022/1/19 16:30
    @author:wangfc
    @version:
    @description:
    借鉴  rasa.nlu.extractor.regex_entity_extrator.RegexEntityExtractor
    实现正则的分类方法

    @params:
    @return:
    """
    defaults = {
        # text will be processed with case insensitive as default
        "case_sensitive": False,
        # use regexes to extract entities
        "use_regexes": True,
        "predict_intent_confidence": 1.0
    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 pattern_data: Optional[PatternData] = None,
                 check_for_contradiction_regex=False):
        super(IntentAttributeRegexClassifier, self).__init__(component_config)
        # 加载正则表达式
        self.pattern_data_dir = self.component_config.get("pattern_data_dir", None)
        self.case_sensitive = self.component_config.get("case_sensitive", False)

        if self.pattern_data_dir:
            pattern_data = PatternData(data_dir=os.path.dirname(self.pattern_data_dir))
        self.pattern_data = pattern_data

        # self.patterns = patterns or self._load_pattern_from_file(self.regex_path)

        self.check_for_contradiction_regex = check_for_contradiction_regex
        if self.pattern_data and self.check_for_contradiction_regex:
            self._check_for_contradiction_regex()





    def _check_for_contradiction_regex(self):
        """
        自动检查 是否存在 正则冲突的情况：
        正则是否存在包含关系？如：开通.{1,3}板账户 vs 开通(创业板)?账户

        使用 https://github.com/qntm/greenery
        """
        from greenery.lego import parse
        pattern_size = self.patterns.__len__()
        check_results = []
        patterns = [pattern[PATTERN] for pattern in self.patterns]
        parse_patterns= [parse(pattern) for pattern in patterns]
        for pattern_a_index in range(0,pattern_size-1):
            for pattern_b_index in range(pattern_a_index+1,pattern_size):
                start_time = time.time()
                prase_pattern_a = parse_patterns[pattern_a_index]
                prase_pattern_b = parse_patterns[pattern_b_index]
                intersection = prase_pattern_a & prase_pattern_b
                if intersection:
                    pattern_a = patterns[pattern_a_index]
                    pattern_b = patterns[pattern_b_index]
                    intersection_message = f"pattern_a={pattern_a},pattern_b={pattern_b} 存在intersection={intersection} "
                    logger.debug(intersection_message)
                else:
                    intersection_message = None
                print(f"pattern_a_index={pattern_a_index},pattern_b_index={pattern_b_index},cost_time= {time.time()-start_time}")
                check_results.append(intersection_message)

        intersection_messages = [intersection_message for intersection_message in check_results if intersection_message]
        if intersection_messages and  intersection_messages.__len__()>0:
            message="\n".join(intersection_messages)
            raise InvalidRuleException(message=f"正则表达式发生冲突异常，请检查正则表达式:{message}")
            return intersection_messages
        else:
            return False


    def train(
            self,
            training_data: TrainingData,
            config: Optional[NLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:

        # self.patterns = pattern_utils.extract_patterns(
        #     training_data,
        #     use_lookup_tables=self.component_config["use_lookup_tables"],
        #     use_regexes=self.component_config["use_regexes"],
        #     use_only_entities=True,
        #     use_word_boundaries=self.component_config["use_word_boundaries"],
        # )

        # 从正则的配置文件中读取 pattern
        if not self.patterns:
            raise_warning(
                "No lookup tables or regexes defined in the training data that have "
                "a name equal to any entity in the training data. In order for this "
                "component to work you need to define valid lookup tables or regexes "
                "in the training data."
            )

    def process(self, message: Message, pattern_data:Optional[PatternData]=None, **kwargs: Any) -> None:

        pattern_data = pattern_data if pattern_data else self.pattern_data
        if not pattern_data :
            return
        # extracted_entities = self._extract_entities(message)
        # extracted_entities = self.add_extractor_name(extracted_entities)
        #
        # message.set(
        #     ENTITIES, message.get(ENTITIES, []) + extracted_entities, add_to_output=True
        # )

        matched_result = self._classify(message,pattern_data)
        if matched_result:
            matched_result = self._add_component_name(matched_result)
            # 增加 INTENT_AND_ATTRIBUTE 属性
            message.set(
                INTENT_AND_ATTRIBUTE, matched_result, add_to_output=True
            )
            return matched_result

    def _classify(self, message: Message,pattern_data:PatternData) -> Dict[Text, Union[Text, float]]:
        """Extract entities of the given type from the given user message."""
        matched_result = {}

        flags = 0  # default flag
        if not self.case_sensitive:
            flags = re.IGNORECASE

        for pattern in pattern_data.patterns:
            try:
                match = re.match(pattern=pattern.pattern, string=message.get(TEXT), flags=flags)
                if not match:
                    raw_text_match = re.match(pattern=pattern.pattern, string=message.get(RAW_TEXT), flags=flags)
                if match or raw_text_match:
                    intent = pattern.intent
                    attribute = pattern.attribute
                    matched_result = {PREDICT_INTENT: intent, PREDICT_INTENT_CONFIDENCE: 1.0,
                                      PREDICT_ATTRIBUTE: attribute, PREDICT_ATTRIBUTE_CONFIDENCE: 1.0}
                    break
            except Exception as e:
                logger.warning(f"pattern={pattern}匹配发生错误：{e}")
            # matches = re.finditer(pattern["pattern"], message.get(TEXT), flags=flags)
            # matches = list(matches)
            # for match in matches:
            #     start_index = match.start()
            #     end_index = match.end()
            #     entities.append(
            #         {
            #             ENTITY_ATTRIBUTE_TYPE: pattern["name"],
            #             ENTITY_ATTRIBUTE_START: start_index,
            #             ENTITY_ATTRIBUTE_END: end_index,
            #             ENTITY_ATTRIBUTE_VALUE: message.get(TEXT)[
            #                                     start_index:end_index
            #                                     ],
            #         }
            #     )
        return matched_result

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Text,
            model_metadata: Optional["Metadata"] = None,
            cached_component: Optional["RegexClassifier"] = None,
            **kwargs: Any,
    ) -> "RegexClassifier":
        """Loads trained component (see parent class for full docstring)."""
        file_name = meta.get("file")
        regex_file = os.path.join(model_dir, file_name)

        if os.path.exists(regex_file):
            patterns = read_json_file(regex_file)
            return IntentAttributeRegexClassifier(meta, patterns=patterns)

        return IntentAttributeRegexClassifier(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        file_name = f"{file_name}.json"
        regex_file = os.path.join(model_dir, file_name)
        dump_obj_as_json_to_file(regex_file, self.patterns)

        return {"file": file_name}

