#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/19 15:59 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/19 15:59   wangfc      1.0         None
"""
import sys
if sys.version >="3.7.0":
    from typing import List, Text,OrderedDict
else:
    from typing import List, Text
    from collections import OrderedDict

from utils.io import read_from_yaml
import logging
logger  = logging.getLogger(__name__)

def read_financial_terms(financial_terms_path):
    """
    @time:  2021/7/19 15:58
    @author:wangfc
    @version:
    @description: 读取 yaml 格式的金融专业词汇

    @params:
    @return:
    """
    financial_terms_yaml = read_from_yaml(path=financial_terms_path)
    financial_terms_set = set()
    # 解析 金融专业词汇
    assert isinstance(financial_terms_yaml, OrderedDict)

    for domain,financial_term_ls in financial_terms_yaml.items():
        # print(f"读取 domain={domain}")
        assert isinstance(financial_term_ls,List)
        # 每个金融专业词汇可能是 简单的 text,或者包含简称等的字典
        for term in financial_term_ls:
            if isinstance(term, Text):
                financial_terms_set.add(term)
            elif isinstance(term, OrderedDict):
                for term_key,term_value in term.items():
                    financial_terms_set.add(term_key)
                    # 对于每个 term 的 解释是字典类型 ：如 简称：***
                    if isinstance(term_value,OrderedDict):
                        for k,v in term_value.items():
                            financial_terms_set.add(v)
                    # 对于每个 term 的 解释是列表类型 ：如 近义词
                    elif isinstance(term_value,List):
                        for t in term_value:
                            if isinstance(t,Text):
                                financial_terms_set.add(t)
    sorted_financial_terms =  sorted(financial_terms_set)
    logger.info(f"共读取 {sorted_financial_terms.__len__()} 个金融专业词汇 from {financial_terms_path}")
    return sorted_financial_terms

if __name__ == '__main__':
    import os
    financial_terms_path = os.path.join('data', 'finance', 'financial_term.yml')
    financial_terms_set =  read_financial_terms(financial_terms_path=financial_terms_path)