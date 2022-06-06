#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/22 14:06 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/22 14:06   wangfc      1.0         None
"""
import sys
sys.path.append('/home/wangfc/faq')
from utils.time import  timeit

@timeit
def test():
    print('test_package')

if __name__ == '__main__':
    test()