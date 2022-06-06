#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/8/8 10:46 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/8 10:46   wangfc      1.0         None
"""

from pathlib import Path

import os
from utils.io import read_yaml_file


class RegxDataSet():
    def __init__(self,regx_path:Path):
        self.regx_path = regx_path
        self.advice_name = 'advice'
        self.faq_name = 'faq'
        self.chat_name = 'chat'

        self.regx_data = self.read_regx_data()
        self.advice_regx_data = self.regx_data.get(self.advice_name)


    def read_regx_data(self):
        regx_data = read_yaml_file(self.regx_path)
        return regx_data

if __name__ == '__main__':
    regx_path = os.path.join('corpus','haitong_intent_classification_data','intent_regx.yml')
    regx_dataset = RegxDataSet(regx_path=regx_path)