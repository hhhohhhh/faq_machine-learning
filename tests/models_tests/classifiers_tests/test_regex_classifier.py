#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/24 9:38 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/24 9:38   wangfc      1.0         None
"""

import os
from typing import List, Text

from data_process.training_data.message import Message
from hsnlp_faq_utils.preprocess.word_segement import HSNLPWordSegmentApi
from models.classifiers.regex_classifiers import IntentAttributeRegexClassifier
from utils.constants import TEXT
from greenery.lego import parse



def test_load_ymal_file():
    regex_path = os.path.join('corpus','hsnlp_kbqa_faq_data','regex_data',"intent_classifier_regex.yml")
    component_config = {"regex_path":regex_path}
    intent_attribute_regex_classifier = IntentAttributeRegexClassifier(component_config=component_config)
    print(intent_attribute_regex_classifier.patterns)
    assert isinstance(intent_attribute_regex_classifier.patterns,list)


def test_check_for_contradiction_regex_tuple(regex_pattern_tuple):
    check_results = []
    for pattern_01,pattern_02 in regex_pattern_tuple:
        intersection = parse(pattern_01) & parse(pattern_02)
        if intersection:
            check_result =  True
            print(f"pattern_01={pattern_01},pattern_02={pattern_02} 存在intersection={intersection} ")
        else:
            check_result =  False
        check_results.append(check_result)
    return check_results

def test_check_for_contradiction_regex(regex_test_data):
    regex_path = os.path.join('corpus','hsnlp_kbqa_faq_data','regex_data',"intent_classifier_regex.yml")
    component_config = {"regex_path":regex_path}
    intent_attribute_regex_classifier = IntentAttributeRegexClassifier(component_config=component_config)
    assert intent_attribute_regex_classifier is not None



def test_regex_classifier(regex_test_data):
    """
    验证 开户 + 开户操作 的数据
    --fixtures ./tests/apps_tests/intent_attribute_classifier_apps_tests
    """
    regex_path = os.path.join('corpus','hsnlp_kbqa_faq_data','regex_data',"intent_classifier_regex.yml")
    component_config = {"regex_path":regex_path}
    intent_attribute_regex_classifier = IntentAttributeRegexClassifier(component_config=component_config)
    if intent_attribute_regex_classifier.patterns:
        for pattern in intent_attribute_regex_classifier.patterns:
            print(pattern)
    results = []
    # intent_account_operation_data = []
    hsnlp_word_segment_url = "http://10.20.33.3:8017/hsnlp/faq/sentenceProcess"
    hsnlp_word_segment_api = HSNLPWordSegmentApi(url=hsnlp_word_segment_url)

    for intent_and_attribute,data in regex_test_data.items():
        for text in data:
            processed_text = hsnlp_word_segment_api.word_segment(text=text)
            data = {TEXT:processed_text}

            message= Message(data=data)
            result = intent_attribute_regex_classifier.process(message=message)
            if result is None:
                print(f"Not working intent_and_attribute= {intent_and_attribute}: {processed_text}")
            results.append(result)

    assert isinstance(results,list)

