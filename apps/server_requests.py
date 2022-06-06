#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/8 11:15 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/8 11:15   wangfc      1.0         None
"""
import json
import ast
from typing import Dict, Text, Tuple

import requests

from rasa.shared.nlu.constants import OUTPUT_INTENT, INTENT, PREDICTED_CONFIDENCE_KEY, INTENT_NAME_KEY, ENTITIES, \
    ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_VALUE, TEXT, ENTITY_ATTRIBUTE_START, ENTITY_ATTRIBUTE_END

import logging

logger = logging.getLogger(__name__)



def post_data(url: Text,
              data: Dict[Text, Text]=None,
              json_data: Dict[Text, Text]=None):
    """
    @time:  2021/10/8 11:16
    @author:wangfc
    @version:
    @description:


    @params:
    @return:
    """
    if data:
        response= requests.post(url, data=data)
    elif json:
        # post方法提交 json 数据,注意 这儿 json 数据直接使用 dict对象传入，而不需要转换为 json 对象
        response = requests.post(url, json=json_data)


    if response.status_code == 400:
        raise ValueError(f"data={data},返回错误")
    try:
        response_data= json.loads(response.text)
    except json.decoder.JSONDecodeError as e:
        response_data = ast.literal_eval(response.text)
    except Exception as e:
        raise e
    return response_data



def post_rasa_data(index,question,url):
    data = {
        "sender": f"test_{index}",
        "message": question
    }
    results = post_data(url=url, json_data=data)
    return results

# def parse_rasa_post_result(results, key='text',parse_intent=False):
#     if parse_intent:


def parse_rasa_intent_result(results,
                 output_intent_key =INTENT, # OUTPUT_INTENT,
                 intent_name_key=INTENT_NAME_KEY,
                 confidence_key = PREDICTED_CONFIDENCE_KEY,
                 default_intent='其他',
                 confidence_threshold= 0) ->Tuple[Text,float]:
    try:
        if isinstance(results,list):
            assert results.__len__() ==1
            result = results[0]
        elif isinstance(results,dict):
            result = results

        output_intent_info = result.get(output_intent_key)
        output_intent = output_intent_info.get(intent_name_key)
        output_intent_confidence = output_intent_info.get(confidence_key)
        if output_intent_confidence < confidence_threshold:
            output_intent = default_intent
    except:
        output_intent =  default_intent
        output_intent_confidence = confidence_threshold

    return output_intent,output_intent_confidence


def parse_rasa_attribute(intent):
    pred_intent_split = intent.split("_")
    if pred_intent_split.__len__() == 2:
        _, attribute = pred_intent_split
    else:
        attribute = None
    return attribute


def parse_rasa_entity_result(question,results)-> Dict[Text,Text]:
    entity_results_dict = {}
    try:
        entities = results.get(ENTITIES)
        for index,entity in enumerate(entities):
            entity_type = entity.get(ENTITY_ATTRIBUTE_TYPE)
            # entity_value = entity.get(ENTITY_ATTRIBUTE_VALUE)
            entity_start = entity.get(ENTITY_ATTRIBUTE_START)
            entity_end = entity.get(ENTITY_ATTRIBUTE_END)
            entity_value = question[entity_start:entity_end]
            entity_results_dict.update({f"predict_entity_type_{index}": entity_type,
                                   f"predict_entity_value_{index}":entity_value})
    except Exception as e:
        logger.error(f"解析实体信息错误：{results}:{e}")
    return entity_results_dict


def parse_rasa_nlu_result(results):
    output_intent,output_intent_confidence = parse_rasa_intent_result(results)
    entity_results_dict = parse_rasa_entity_result(results)

    # 模型的意图为 大意图 + 属性
    pred_intent_splited = output_intent.split("-")
    if pred_intent_splited.__len__() == 1:
        intent = pred_intent_splited[0]
        attribute = None
    elif pred_intent_splited.__len__() == 2:
        intent = pred_intent_splited[0]
        attribute = pred_intent_splited[1]
    else:
        logger.error(f"{output_intent}分割后大于2个值，存在错误")
        intent = None
        attribute= None

    nlu_result_dict = entity_results_dict
    nlu_result_dict.update({"predict_intent":intent,'attribute':attribute,})
    return nlu_result_dict




def test_rase_rest_server():
    url = 'http://10.20.33.3:5005/webhooks/rest/webhook'
    data = {
        "sender": "test_user",
        "message": "Hi there!"
    }
    response_dict = post_data(url=url, data=data)


if __name__ == '__main__':
    test_rase_rest_server()
