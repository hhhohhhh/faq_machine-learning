#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/4/8 14:03 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/4/8 14:03   wangfc      1.0         None

"""
import os
from collections import OrderedDict
from pathlib import Path
from ast import literal_eval
args_path = Path().joinpath(Path(__file__).parent,'command_arguments')
kwargs = OrderedDict()
with open(args_path) as f:
    lines =f.readlines()
    for line in lines:
        line = line.replace('\\\n','')
        line = line.strip()
        line_splited = line.split(' ')
        for i in range(int(len(line_splited)/2)):
            key_index = i*2
            value_index = i*2+1
            key = line_splited[key_index].strip().lstrip('-').replace('-', '_')
            # if key[:2] == '--':
            #     key=key[2:]
            # else:
            #     key = key[1:]
            value = line_splited[value_index].strip()
            print(f'key={key},value={value}')
            try:
                value = literal_eval(value)
            except:
                pass
            kwargs.update({key:value})

dict(kwargs)
# def change_type(value):
#     literal_eval('True')
#     if value =='True':
#         return True
#     elif value =='False':
#         return False
#     try:
#         value = int(value)
#         return  value
#     except Exception as e:

