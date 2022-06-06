#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/18 13:39 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/18 13:39   wangfc      1.0         None
"""


def func():
    print("Python is " + x)
    return x

def func_same_variable_name():
    x = "fantastic"
    print("Python is " + x)
    return x


def func_with_global_keyword():
    # If you use the global keyword, the variable belongs to the global scope:

    global x
    sentence1= "Python is " + x
    # Also, use the global keyword if you want to change a global variable inside a function.
    x = 'fantastic'
    global_x = get_global_variable()
    return sentence1,global_x

def get_global_variable():
    global x
    return x


def test_myfunc():
    result = func()
    assert  result == "awesome"
    assert func_same_variable_name() == 'fantastic'
    assert func_with_global_keyword() == ('Python is awesome', 'fantastic')

if __name__ == '__main__':
    x = "awesome"
    func()