#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/19 15:08 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 15:08   wangfc      1.0         None
"""
from typing import Dict,Text,Any

from models.components import Component
from utils.constants import CLASSIFIER



class IntentClassifier(Component):
    pass
    # def add_classifier_name(
    #         self, pred_result: Dict[Text, Any]) -> Dict[Text, Any]:
    #     """Adds this extractor's name to a list of entities.
    #     模仿 EntityExtractor 方法，增加 分类器的名称
    #
    #     Args:
    #         entities: the extracted entities.
    #
    #     Returns:
    #         the modified entities.
    #     """
    #     pred_result[CLASSIFIER] = self.name
    #     return pred_result
