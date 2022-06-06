#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/15 23:07 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/15 23:07   wangfc      1.0         None
"""
from pathlib import Path
from typing import Text, Union


class RetrievalAgent():

    def __init__(self,document_path:Union[Text,Path],model_dir:Union[Text,Path]):
        self.document_path = document_path
        self.model_dir = model_dir
        self.documents = None
        self.index = None

    def _get_documents(self):
        raise NotImplementedError

    def build_index(self):
        raise NotImplementedError

    def load_index(self):
        raise NotImplementedError

    def search(self,query:Text):
        raise NotImplementedError





