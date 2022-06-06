#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/4/12 10:01 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/4/12 10:01   wangfc      1.0         None

"""
from parlai.scripts.display_data import DisplayData

if __name__ == '__main__':

    tasks = ['convai2', 'dstc7', 'ubutu2']
    task = 'convai2:both:2'
    # task = f'{task}:{teacher_class}:{options}'
    task = 'dstc7'
    kwargs = {'task': task}
    DisplayData.main(**kwargs)