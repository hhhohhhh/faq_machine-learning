#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/18 11:20 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/18 11:20   wangfc      1.0         None
"""

def fib_generator(threshold,fib_num=None):
    """
    函数是顺序执行，遇到 return语句或者最后一行函数语句就返回。
    而变成generator的函数，在每次调用next()的时候执行，遇到yield语句返回，再次执行时从上次返回的yield语句处继续执行 (可以记住上次运行的状态)。

    threshold:获取阈值内所有的 fib 数
    fib_num: 只输出 第 fib_num 个 fib 数
    """
    n= 0
    a = 0
    b = 1
    # 当 a+b 小于阈值的时候
    while b < threshold:
        if fib_num:
            if fib_num==n:
                yield b
                return 'None'
        else:
            yield b

        n+=1
        c = a
        a = b
        b = c + b
    return 'Done'


def test_fib_generator():
    # 生成器需要先调用next()迭代一次或者是先send(None)启动
    g = fib_generator(10)
    assert  next(g) ==1
    assert g.send(None) ==1

    # 获取阈值内所有的 fib 数
    fibs = []
    for f in fib_generator(10):
        fibs.append(f)

    assert fibs ==[1,1,2,3,5,8]

    fibs = [f for f in fib_generator(10,5)]
    assert fibs == [8]


def gen_one():
    """
    yield from 是Python 3.3 新引入的语法（PEP 380）。它主要解决的就是在生成器里玩生成器不方便的问题。它有两大主要功能。
    第一个功能是：让嵌套生成器不必通过循环迭代yield，而是直接yield from。

    以下两种在生成器里玩子生成器的方式是等价的:  gen_one == gen_two
    作者：SeanCheney
    链接：https://www.jianshu.com/p/fe146f9781d2

    """
    subgen = range(10)
    yield from subgen

def gen_two():
    subgen = range(10)
    for item in subgen:
        yield item

def test_gen_one():
    for x in gen_one():
        print(x)


