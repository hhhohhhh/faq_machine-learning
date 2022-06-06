#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/8/10 17:12 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/10 17:12   wangfc      1.0         None


"""

import pytest


def test_some_data(some_data):
    assert some_data==42


def test_fixture(fixture_starter):
    print("结束")
    assert fixture_starter ==1
"""





# content of test_class.py
class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    # def test_two(self):
    #     x = "hello"
    #     assert hasattr(x, "check")

"""


"""
很多时候需要在测试前进行预处理（如新建数据库连接），并在测试完成进行清理（关闭数据库连接）。
当有大量重复的这类操作，最佳实践是使用固件来自动化所有预处理和后处理。
Pytest 使用 yield 关键词将固件分为两部分，yield 之前的代码属于预处理，会在测试前执行；yield 之后的代码属于后处理，将在测试完成后执行。
以下测试模拟数据库查询，使用固件来模拟数据库的连接关闭：
"""

"""

@pytest.fixture()
def db():
    print('Connection successful')

    yield

    print('Connection closed')


def search_user(user_id):
    d = {
        '001': 'xiaoming'
    }
    return d[user_id]


def test_search(db):
    assert search_user('001') == 'xiaoming'
"""

