#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/24 14:38 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/24 14:38   wangfc      1.0         None
"""
from typing import Text, List, Union

import requests


class HSNLPWordSegmentApi():
    def __init__(self,url="http://10.20.33.3:8017/hsnlp/faq/sentenceProcess"):
        self.url = url

    def word_segment(self,text:Text,if_get_processed_data=True)-> Union[Text,List[Text]]:
        data = {"sentence": text, "domainId": 1}
        head = {"Connection": "close"}
        r = requests.post(self.url, data=data, headers=head)
        rows = r.json()["baseInfo"][0]
        words = rows["replaceSynonym"]
        # print("word1", words)
        if if_get_processed_data:
            return "".join(words)
        else:
            return words
