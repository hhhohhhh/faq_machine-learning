#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/20 9:05 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/20 9:05   wangfc      1.0         None
"""
import logging
logger = logging.getLogger(__name__)

logger.info("加载模型分类模块")

import datetime
from typing import Dict, Text, Any, Optional

from apps.app_constants import PREPROCESSED_TEXT_KEY, PREDICT_KEY
from apps.intent_attribute_classifier_apps.intent_attribute_classifier_app_constants import PREDICT_INTENT, \
    PREDICT_INTENT_CONFIDENCE, PREDICT_ATTRIBUTE, PREDICT_ATTRIBUTE_CONFIDENCE
from apps.sanic_apps.channel import UserMessage
from data_process.dataset.hsnlp_faq_knowledge_dataset import ATTRIBUTE
from data_process.training_data.message import Message
from hsnlp_faq_utils.preprocess.word_segement import HSNLPWordSegmentApi
from models.classifiers.regex_classifiers import IntentAttributeRegexClassifier, PatternData
from models.model import Interpreter
from utils.constants import TEXT, INTENT_AND_ATTRIBUTE, INTENT, INTENT_NAME_KEY, PREDICTED_CONFIDENCE_KEY, RAW_TEXT, ID, \
    ENTITIES


class IntentAttributeInterpreter(Interpreter):

    def __init__(self,use_intent_attribute_regex_classifier=True,
                 hsnlp_word_segment_api:HSNLPWordSegmentApi=None,
                 from_tfv1=False,
                 pattern_data_dir=None,
                 model_dir=None,run_eagerly=False,
                 default_intent='其他',
                 default_attribute='其他',
                 intent_confidence_threshold=0.9,
                 attribute_confidence_threshold=0.9
                 ):
        """
        重写 初始化函数，相对更加简单一些
        """
        self.hsnlp_word_segment_api = hsnlp_word_segment_api

        self.use_intent_attribute_regex_classifier = use_intent_attribute_regex_classifier
        self.from_tfv1 = from_tfv1
        self.model_dir = model_dir
        self.run_eagerly = run_eagerly
        self.default_intent = default_intent
        self.default_attribute = default_attribute
        self.intent_confidence_threshold = intent_confidence_threshold
        self.attribute_confidence_threshold = attribute_confidence_threshold

        # 加载 正则模块
        self.pattern_data_dir = pattern_data_dir
        regex_component_config = {"pattern_data_dir": self.pattern_data_dir}
        self.intent_attribute_regex_classifier = IntentAttributeRegexClassifier(component_config=regex_component_config)


        # logger.info("加载模型分类模块")
        if self.from_tfv1:
            from apps.intent_attribute_classifier_apps.intent_attribute_classifier import IntentAttributeClassifierTfv1
            interpreter = IntentAttributeClassifierTfv1()
        else:
            from models.model import load_interpreter_from_model
            interpreter = load_interpreter_from_model(model_dir=model_dir,use_compressed_model=False,
                                                      run_eagerly = run_eagerly)
        self.interpreter = interpreter




    @staticmethod
    def default_output_attributes() -> Dict[Text, Any]:
        # from rasa.shared.nlu.constants import INTENT
        return {
            TEXT: "",
            # INTENT_AND_ATTRIBUTE:{PREDICT_INTENT:None,PREDICT_INTENT_CONFIDENCE:0.0,
            #                       PREDICT_ATTRIBUTE: None,PREDICT_ATTRIBUTE_CONFIDENCE:0.0}
            # INTENT: {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
        }

    async def async_handle_message(self,message:UserMessage,
                                   pattern_data:Optional[PatternData]=None,
                                   preprocessed_key="replaceSynonym",)->Dict[Text,Any]:
        """
        @time:  2022/1/20 10:22
        @author:wangfc
        @version:
        @description:
            新增 可选参数 pattern_data，用于服务实时更新正则表达式的时候更新服务

        @params:
        @return:
        """
        text = message.text
        processed_text = None
        if message.metadata:
            # 如果存在预处理的数据，直接获取预处理后的文本
            preprocessed_text_ls = message.metadata.get(preprocessed_key)
            if preprocessed_text_ls and isinstance(preprocessed_text_ls,list):
                processed_text = "".join(preprocessed_text_ls)
        if processed_text:
            pass
        elif processed_text is None and self.hsnlp_word_segment_api:
            # 使用预处理接口获取预处理文本
            processed_text = self.hsnlp_word_segment_api.word_segment(text=text)
        else:
            # 如果预处理的文本为空，并且没有预处理接口，直接将原始文本当做预处理后的文本
            processed_text = text
        intent_and_attribute_prediction = self.parse(text=processed_text,raw_text=text, pattern_data=pattern_data)

        output= {TEXT:text,PREPROCESSED_TEXT_KEY:processed_text,
                 PREDICT_KEY:intent_and_attribute_prediction}
        return output

    def parse(
        self,
        text: Text,
        raw_text:Optional[Text] =None,
        pattern_data: Optional[PatternData] = None,
        time: Optional[datetime.datetime] = None,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:

        timestamp = int(time.timestamp()) if time else None

        if not text:
            # Not all components are able to handle empty strings. So we need
            # to prevent that... This default return will not contain all
            # output attributes of all components, but in the end, no one
            # should pass an empty string in the first place.
            # output = self.default_output_attributes()
            # output["text"] = ""
            # output.update({PREPROCESSED_TEXT_KEY: text})
            intent_and_attribute_prediction = {PREDICT_INTENT:self.default_intent,
                                               PREDICT_INTENT_CONFIDENCE: 1.0,
                                               PREDICT_ATTRIBUTE:self.default_attribute,
                                               PREDICT_ATTRIBUTE_CONFIDENCE: 1.0}
            return intent_and_attribute_prediction

        else:

            data = self.default_output_attributes()
            data[TEXT] = text
            data[RAW_TEXT] = raw_text
            # output_properties: 表示最后输出的属性
            message = Message(data=data, time=timestamp,output_properties= {INTENT_AND_ATTRIBUTE})
            # for component in self.pipeline:
            #     component.process(message, **self.context)
            #
            # if not self.has_already_warned_of_overlapping_entities:
            #     self.warn_of_overlapping_entities(message)

            # 是否使用 正则预测意图和属性
            matched_result = None
            if self.use_intent_attribute_regex_classifier and self.intent_attribute_regex_classifier:
                matched_result = self.intent_attribute_regex_classifier.process(message=message,pattern_data=pattern_data)

            # 判断是否已经存在意图和属性
            if matched_result is None:
                # self.intent_attribute_classifier.process(message=message)
                self._interpreter_process(message=message)

            output = self.default_output_attributes()
            output.update(message.as_dict(only_output_properties=only_output_properties))
            # 最终只输出 intent_and_attribute
            intent_and_attribute_prediction = output.get(INTENT_AND_ATTRIBUTE)
            # 使用阈值进行过滤
            intent_and_attribute_prediction = self._output_by_confidence_threshold(intent_and_attribute_prediction)
            # 账户查询修改为查询账户
            intent_and_attribute_prediction = self._change_attribute_name(intent_and_attribute_prediction)

            # 增加entities 输出
            output_pred = intent_and_attribute_prediction.copy()
            entities_prediction = output.get(ENTITIES)
            if entities_prediction:
                output_pred.update({ENTITIES:entities_prediction})

            return output_pred

    def _interpreter_process(self,message:Message=None,from_tfv1=True):
        result = self.interpreter.parse(message=message)
        output_result = self._output_as_predict_label_to_confidence_dict(result)
        message.set(prop=INTENT_AND_ATTRIBUTE, info=output_result, add_to_output=True)

    def _output_as_predict_label_to_confidence_dict(self,result:Dict[Text,Dict[Text,Any]]):
        intent_info = result.get(INTENT,{})
        attribute_info = result.get(ATTRIBUTE,{})
        if ID in intent_info:
            intent_info.pop(ID)
        if ID in attribute_info:
            attribute_info.pop(ID)
        output_result = {}
        if INTENT_NAME_KEY in intent_info:
            intent = intent_info.get(INTENT_NAME_KEY)
            intent_info.pop(INTENT_NAME_KEY)
            # intent_info.update({PREDICT_INTENT: intent})
            output_result.update({PREDICT_INTENT: intent})

        if PREDICTED_CONFIDENCE_KEY in intent_info:
            intent_confidence = intent_info.get(PREDICTED_CONFIDENCE_KEY)
            intent_info.pop(PREDICTED_CONFIDENCE_KEY)
            # intent_info.update({PREDICT_INTENT_CONFIDENCE: intent_confidence})
            output_result.update({PREDICT_INTENT_CONFIDENCE: intent_confidence})
        output_result.update(intent_info)

        if INTENT_NAME_KEY in attribute_info:
            attribute = attribute_info.get(INTENT_NAME_KEY)
            attribute_info.pop(INTENT_NAME_KEY)
            # attribute_info.update({PREDICT_ATTRIBUTE: attribute})
            output_result.update({PREDICT_ATTRIBUTE: attribute})

        if PREDICTED_CONFIDENCE_KEY in attribute_info:
            attribute_confidence = attribute_info.get(PREDICTED_CONFIDENCE_KEY)
            attribute_info.pop(PREDICTED_CONFIDENCE_KEY)
            # attribute_info.update({PREDICT_ATTRIBUTE_CONFIDENCE: attribute_confidence})
            output_result.update({PREDICT_ATTRIBUTE_CONFIDENCE: attribute_confidence})
        output_result.update(attribute_info)

        # intent = result.get(INTENT,{}).get(INTENT_NAME_KEY)
        # intent_confidence  = result.get(INTENT,{}).get(PREDICTED_CONFIDENCE_KEY)
        # attribute = result.get(ATTRIBUTE,{}).get(INTENT_NAME_KEY)
        # attribute_confidence = result.get(ATTRIBUTE, {}).get(PREDICTED_CONFIDENCE_KEY)
        # result = {PREDICT_INTENT: intent, PREDICT_INTENT_CONFIDENCE: intent_confidence,
        #           PREDICT_ATTRIBUTE: attribute, PREDICT_ATTRIBUTE_CONFIDENCE: attribute_confidence
        #           }
        return output_result

    def _output_by_confidence_threshold(self,intent_and_attribute_prediction:Dict):
        predict_intent_confidence = intent_and_attribute_prediction.get(PREDICT_INTENT_CONFIDENCE)
        predict_attribute_confidence = intent_and_attribute_prediction.get(PREDICT_ATTRIBUTE_CONFIDENCE)
        if self.intent_confidence_threshold:
            if predict_intent_confidence is None or predict_intent_confidence < self.intent_confidence_threshold:
                intent_and_attribute_prediction.update({PREDICT_INTENT:self.default_intent})
        if self.attribute_confidence_threshold:
            if predict_attribute_confidence is None or predict_attribute_confidence < self.attribute_confidence_threshold:
                intent_and_attribute_prediction.update({PREDICT_ATTRIBUTE:self.default_attribute})
        return intent_and_attribute_prediction

    def _change_attribute_name(self,intent_and_attribute_prediction:Dict):
        """
        账户查询 修改为 查询账户
        """
        predict_attribute = intent_and_attribute_prediction.get(PREDICT_ATTRIBUTE)
        if predict_attribute=='账户查询':
            intent_and_attribute_prediction.update({PREDICT_ATTRIBUTE:"查询账户"})
        return intent_and_attribute_prediction