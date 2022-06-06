#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/19 11:11 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 11:11   wangfc      1.0         None
"""
class Demo():
    """
    当我们创建一个实例的时候，实际上是先调用的__new__函数创建实例，然后再调用__init__对实例进行的初始化。
    __init__并不是构造函数，它只是初始化方法。也就是说在调用__init__之前，我们的实例就已经被创建好了，__init__只是为这个实例赋上了一些值

    """