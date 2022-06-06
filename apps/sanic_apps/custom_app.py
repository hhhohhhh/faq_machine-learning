#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/24 15:14 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/24 15:14   wangfc      1.0         None
"""
from typing import Optional, Any

from sanic import Sanic
from types import SimpleNamespace


class CustomSanic(Sanic):
    """
    sanic 21.0 版本只支持 Python 3.7.0+版本，因此 自定义CustomSanic 对象，增加 context 属性
    """
    def __init__(self,ctx:Optional[Any]=None,*args,**kwargs):
        super(CustomSanic,self).__init__(*args,**kwargs)
        self.ctx = ctx or SimpleNamespace()
