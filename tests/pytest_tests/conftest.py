#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/8/11 8:38 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/11 8:38   wangfc      1.0         None


在使用中如果发现需要使用多个文件中的fixture，则可以将fixture写入 conftest.py中。
使用时不需要导入要在测试中使用的fixture，它会自动被 pytest 发现。

"""
import pytest


@pytest.fixture()
def fixture_starter():
    print("fixture-starter开始")
    return 1

# @pytest.fixture()装饰器用于声明函数是一个fixture。如果测试函数的参数列表中包含fixture名，
# 那么pytest会检测到，并在测试函数运行之前执行fixture。
@pytest.fixture()
def some_data():
    return 42