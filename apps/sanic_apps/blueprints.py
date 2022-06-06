#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/14 9:42 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/14 9:42   wangfc      1.0         None
"""
from typing import Text

from sanic import Sanic
from sanic.response import HTTPResponse, json, text

from apps.sanic_apps.channel import InputChannel


