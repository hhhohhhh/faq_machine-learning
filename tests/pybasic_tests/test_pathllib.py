#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/18 17:15 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/18 17:15   wangfc      1.0         None
"""
import os
import sys
import pathlib
from pathlib import Path

def test_pathlib():
    #
    # 直接传进一个完整字符串
    p = Path('C:/Users/dongh/Documents/python_learn/pathlib_/file1.txt')
    # 也可以传进多个字符串
    p = Path('C:\\', 'Users', 'dongh', 'Documents', 'python_learn', 'pathlib_', 'file1.txt')

    # 用户目录/当前目录
    pathlib.Path.home()
    pathlib.Path.cwd()

    pathlib.Path

    # 获取上级目录
    # cwd = os.getcwd()
    cwd = pathlib.Path.cwd()
    cwd.parent
    cwd.parent.parent
    # 获创建路径
    file_path = cwd.joinpath('my.conf')

    # 文件操作
    file_path.write_text("debug = 1\n")
    with file_path.open(mode='r') as fd:
        for line in fd:
            print(line)
    print(file_path.read_text())




