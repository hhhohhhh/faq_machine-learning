#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/24 10:56 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/24 10:56   wangfc      1.0         None
"""

import os
import re
from typing import Text, List, Dict




# TODO: 将 financial_entity.yml 数据与 financial_term.yml 融合为 financial_knowledge_base.yml ，
#      开发 FinancialKnowledgeBase 可以方便地读取知识库，并且可以转换为 rasa 的那种数据格式
from utils.io import read_yaml_file


class FinancialTerms():
    """
    金融实体词类：加载和处理金融实体
    """
    def __init__(self,financial_term_data_path=None):
        self.examples_key = 'examples'
        if financial_term_data_path is None:
            financial_term_data_path = os.path.join('corpus', 'finance', 'financial_entity.yml')
        self.financial_term_data_path = financial_term_data_path


    def _load_finacial_term(self):

        financial_term_rasa_format_data = read_yaml_file(filename=self.financial_term_data_path)
        return financial_term_rasa_format_data


    def get_financial_term_data(self) -> List[Dict[Text,Text]]:
        """
        将  yaml 中 string 格式的 examples 转换为 list
        """
        financial_term_rasa_format_data = self._load_finacial_term()
        financial_terms_data =  financial_term_rasa_format_data['nlu']
        for item in financial_terms_data:
            examples = []
            examples_out = item.get('examples')
            if isinstance(examples_out,str):
                for ex in examples_out.split('\n'):
                    matched = re.match(pattern=(r'^- {*}?(.*)$'),string=ex)
                    if matched:
                        example = matched.groups()[0]
                        examples.append(example)
            elif isinstance(examples_out,list):
                examples = examples_out
            else:
                raise ValueError(f"item={item}, examples_out 为不支持类型: {type(examples_out)}")

            item.update({'examples':examples})
        return financial_terms_data


    def get_block_synonyms(self,keys=["synonym"],block_values = ['新三板','科创板','创业板','北交所']):
        examples = []
        for key in keys:
            examples.extend(self.get_key_examples_withinvalues(key=key, values=block_values))
        return examples


    def get_key_examples_withinvalues(self,key:Text,values:List[Text]):
        """
        获取 keys 中的所有属于 values的  examples
        """
        examples = values.copy()
        for data in  self.financial_terms_data:
            if data.get(key) and data.get(key) in values:
                examples.extend(data.get('examples'))
        return list(set(examples))


if __name__ == '__main__':
    financial_terms = FinancialTerms()
