#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/29 9:47 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/29 9:47   wangfc      1.0         None
"""

def fun(a=None,b=None,c=None,*args,**kwargs):
    print(f"a={a},b={b},c={c},args={args},kwargs={kwargs}")

def test_argument_from_dict():
    arguments = {'b':10,'d':1}
    fun(**arguments)

    ls = [1,2,3,4]
    fun(*ls)

    fun(None,None,None,ls)

    fun(None, None, None, *ls)



