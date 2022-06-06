#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2020/11/11 9:54 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/11 9:54   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

"""
from collections import defaultdict
d = defaultdict(list)
# 给它赋值，同时也是读取
d['h']
d['h'].append('haha')

words = ['hello', 'world', 'nice', 'world']
#使用lambda来定义简单的函数
counter = defaultdict(lambda: 0)
for kw in words:
    counter[kw] += 1
counter